"""Configuration regressions using real SDK loading and synthetic files only."""

import builtins
import base64
import json
import os
import socket
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml
from kubernetes import client, config
from kubernetes.config.config_exception import ConfigException
from kubernetes.config import kube_config

from kubectl_mcp_tool import k8s_config
from kubectl_mcp_tool.providers import (
    KubernetesProvider,
    ProviderConfig,
    ProviderError,
    ProviderType,
    UnknownContextError,
)


def write_config(path, name="alpha", server="https://alpha.invalid", current="alpha"):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "apiVersion": "v1",
        "kind": "Config",
        "clusters": [{"name": name, "cluster": {"server": server}}],
        "users": [{"name": name, "user": {"token": f"synthetic-{name}"}}],
        "contexts": [
            {
                "name": name,
                "context": {"cluster": name, "user": name, "namespace": name},
            }
        ],
    }
    if current is not None:
        data["current-context"] = current
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def bearer_token(configuration):
    # SDK 37 renamed this key; the configured Authorization header is unchanged.
    return configuration.api_key.get(
        "authorization", configuration.api_key.get("BearerToken")
    )


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    """Redirect defaults, block cluster traffic, and restore singleton/SDK state."""
    synthetic_home = tmp_path / "home"
    default = synthetic_home / ".kube" / "config"
    original_expanduser = os.path.expanduser

    def expanduser(path):
        if str(path).startswith("~/"):
            return str(synthetic_home / str(path)[2:])
        return original_expanduser(path)

    monkeypatch.setattr(os.path, "expanduser", expanduser)
    monkeypatch.setattr(kube_config, "KUBE_CONFIG_DEFAULT_LOCATION", str(default))
    for key in (
        "MCP_K8S_PROVIDER",
        "MCP_K8S_KUBECONFIG",
        "KUBECONFIG",
        "MCP_K8S_CONTEXT",
        "MCP_STATELESS_MODE",
        "MCP_KUBECONFIG_WATCH",
        "KUBERNETES_SERVICE_HOST",
        "KUBERNETES_SERVICE_PORT",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(KubernetesProvider, "_instance", None)
    monkeypatch.setattr(k8s_config, "_stateless_mode", False)
    monkeypatch.setattr(k8s_config, "_config_loaded", False)
    monkeypatch.setattr(k8s_config, "_kubeconfig_watcher", None)
    monkeypatch.setattr(k8s_config, "_config_change_callbacks", [])
    monkeypatch.setattr(client.Configuration, "_default", None)
    in_cluster = Mock(side_effect=ConfigException("synthetic: not in a pod"))
    monkeypatch.setattr(config, "load_incluster_config", in_cluster)
    monkeypatch.setattr(
        socket.socket, "connect", Mock(side_effect=AssertionError("network forbidden"))
    )
    monkeypatch.setattr(
        client.ApiClient,
        "call_api",
        Mock(side_effect=AssertionError("cluster request forbidden")),
    )
    opened_clients = []
    original_init = client.ApiClient.__init__

    def tracked_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        opened_clients.append(self)

    monkeypatch.setattr(client.ApiClient, "__init__", tracked_init)
    yield synthetic_home, in_cluster
    k8s_config.disable_kubeconfig_watch()
    for api_client in opened_clients:
        api_client.close()


@pytest.mark.parametrize("current", [None, ""])
@pytest.mark.parametrize("stateless", [False, True])
def test_no_current_context_lists_and_loads_explicit_client(
    tmp_path, monkeypatch, current, stateless
):
    path = write_config(tmp_path / "config", current=current)
    monkeypatch.setenv("MCP_K8S_KUBECONFIG", str(path))
    k8s_config.set_stateless_mode(stateless)
    assert k8s_config.list_contexts() == [
        {
            "name": "alpha",
            "cluster": "alpha",
            "user": "alpha",
            "namespace": "alpha",
            "active": False,
        }
    ]
    assert k8s_config.get_active_context() is None
    assert (
        k8s_config.get_k8s_client("alpha").api_client.configuration.host
        == "https://alpha.invalid"
    )
    with pytest.raises(ProviderError, match="[Cc]ontext"):
        k8s_config._load_config_for_context()


def test_pathlist_expands_each_path_and_ignores_empty_entries(
    isolated_config, monkeypatch
):
    synthetic_home, in_cluster = isolated_config
    write_config(synthetic_home / "one", current=None)
    write_config(
        synthetic_home / "two",
        name="beta",
        server="https://beta.invalid",
        current="beta",
    )
    pathlist = os.pathsep.join(["", "~/one", "", "~/two", "~/one", ""])
    monkeypatch.setenv("KUBECONFIG", pathlist)
    provider = KubernetesProvider()
    assert [ctx.name for ctx in provider.list_contexts()] == ["alpha", "beta"]
    assert provider.get_current_context() == "beta"
    assert provider.get_api_client("beta").configuration.host == "https://beta.invalid"
    in_cluster.assert_not_called()


def test_merged_first_wins_and_relative_credentials_keep_file_origin(tmp_path):
    first = write_config(tmp_path / "clusters" / "config", current="alpha")
    second = write_config(tmp_path / "users" / "config", name="beta", current="beta")
    a = yaml.safe_load(first.read_text())
    a["clusters"][0]["cluster"]["certificate-authority"] = "ca.pem"
    a["users"] = []
    first.write_text(yaml.safe_dump(a))
    b = yaml.safe_load(second.read_text())
    b["clusters"].append(
        {"name": "alpha", "cluster": {"server": "https://wrong.invalid"}}
    )
    b["contexts"].append(
        {"name": "alpha", "context": {"cluster": "beta", "user": "beta"}}
    )
    b["users"].append(
        {
            "name": "alpha",
            "user": {
                "tokenFile": "token",
                "client-certificate": "client.pem",
                "client-key": "client.key",
            },
        }
    )
    second.write_text(yaml.safe_dump(b))
    (first.parent / "ca.pem").write_text("synthetic-ca")
    (second.parent / "client.pem").write_text("synthetic-cert")
    (second.parent / "client.key").write_text("synthetic-key")
    (second.parent / "token").write_text("synthetic-file-token")
    provider = KubernetesProvider(
        ProviderConfig(kubeconfig_path=os.pathsep.join([str(first), str(second)]))
    )
    assert provider.get_current_context() == "alpha"
    assert [(ctx.name, ctx.cluster) for ctx in provider.list_contexts()] == [
        ("alpha", "alpha"),
        ("beta", "beta"),
    ]
    loaded = provider.get_api_client("alpha").configuration
    assert loaded.host == "https://alpha.invalid"
    assert loaded.ssl_ca_cert == str(first.parent / "ca.pem")
    assert loaded.cert_file == str(second.parent / "client.pem")
    assert loaded.key_file == str(second.parent / "client.key")
    assert bearer_token(loaded) == "Bearer synthetic-file-token"


@pytest.mark.parametrize("source", ["mcp", "environment", "default"])
@pytest.mark.parametrize("stateless", [False, True])
def test_all_entrypoints_use_same_source(
    tmp_path, isolated_config, monkeypatch, source, stateless
):
    synthetic_home, in_cluster = isolated_config
    write_config(synthetic_home / ".kube" / "config", name="default", current="default")
    env = write_config(tmp_path / "env", name="environment", current="environment")
    mcp = write_config(tmp_path / "mcp", name="mcp", current="mcp")
    if source in ("mcp", "environment"):
        monkeypatch.setenv("KUBECONFIG", str(env))
    if source == "mcp":
        monkeypatch.setenv("MCP_K8S_KUBECONFIG", str(mcp))
    k8s_config.set_stateless_mode(stateless)
    assert [ctx["name"] for ctx in k8s_config.list_contexts()] == [source]
    assert k8s_config.get_active_context() == source
    assert k8s_config.context_exists(source)
    assert (
        bearer_token(k8s_config.get_k8s_client(source).api_client.configuration)
        == f"Bearer synthetic-{source}"
    )
    assert k8s_config.load_kubernetes_config(source) is True
    assert (
        bearer_token(client.Configuration.get_default_copy())
        == f"Bearer synthetic-{source}"
    )
    config.load_kube_config()
    assert (
        bearer_token(client.Configuration.get_default_copy())
        == f"Bearer synthetic-{source}"
    )
    in_cluster.assert_not_called()


@pytest.mark.parametrize(
    "entrypoint", ["list", "active", "exists", "client", "load", "patched"]
)
@pytest.mark.parametrize(
    "contents", ["clusters: [", "[]", "", "contexts: wrong\n", "contexts: [{}]\n"]
)
def test_invalid_primary_config_propagates_without_fallback(
    tmp_path, monkeypatch, entrypoint, contents
):
    bad = tmp_path / "bad"
    bad.write_text(contents)
    fallback = write_config(tmp_path / "fallback")
    monkeypatch.setenv("MCP_K8S_KUBECONFIG", str(bad))
    monkeypatch.setenv("KUBECONFIG", str(fallback))
    operations = {
        "list": k8s_config.list_contexts,
        "active": k8s_config.get_active_context,
        "exists": lambda: k8s_config.context_exists("alpha"),
        "client": lambda: k8s_config._load_config_for_context("alpha"),
        "load": lambda: k8s_config.load_kubernetes_config("alpha"),
        "patched": config.load_kube_config,
    }
    with pytest.raises(ProviderError) as error:
        operations[entrypoint]()
    assert str(bad) in str(error.value)
    assert error.value.__cause__ is not None


@pytest.mark.parametrize("stateless", [False, True])
@pytest.mark.parametrize("missing_member", [False, True])
def test_explicit_missing_file_never_uses_incluster(
    tmp_path, isolated_config, monkeypatch, stateless, missing_member
):
    _, in_cluster = isolated_config
    in_cluster.side_effect = None
    missing = tmp_path / "missing"
    path = str(missing)
    if missing_member:
        path = os.pathsep.join([str(write_config(tmp_path / "valid")), path])
    monkeypatch.setenv("MCP_K8S_KUBECONFIG", path)
    k8s_config.set_stateless_mode(stateless)
    with pytest.raises(ProviderError, match="missing"):
        k8s_config._load_config_for_context("alpha")
    in_cluster.assert_not_called()


def test_refresh_error_does_not_return_stale_context_or_client(tmp_path, monkeypatch):
    path = write_config(tmp_path / "config")
    monkeypatch.setenv("MCP_K8S_KUBECONFIG", str(path))
    original = k8s_config._load_config_for_context("alpha")
    path.write_text("clusters: [")
    for operation in (
        k8s_config.list_contexts,
        k8s_config.get_active_context,
        lambda: k8s_config._load_config_for_context("alpha"),
    ):
        with pytest.raises(ProviderError):
            operation()
    write_config(path, server="https://recovered.invalid")
    recovered = k8s_config._load_config_for_context("alpha")
    assert recovered is not original
    assert recovered.configuration.host == "https://recovered.invalid"


@pytest.mark.parametrize("stateless", [False, True])
def test_same_context_updates_server_credentials_and_current_context(
    tmp_path, monkeypatch, stateless
):
    path = write_config(tmp_path / "config")
    monkeypatch.setenv("MCP_K8S_KUBECONFIG", str(path))
    k8s_config.set_stateless_mode(stateless)
    old = k8s_config._load_config_for_context("alpha")
    unchanged = k8s_config._load_config_for_context("alpha")
    assert (old is unchanged) is (not stateless)
    data = yaml.safe_load(path.read_text())
    data["clusters"][0]["cluster"]["server"] = "https://new.invalid"
    data["users"][0]["user"]["token"] = "synthetic-new-token"
    data["contexts"].append(
        {"name": "beta", "context": {"cluster": "alpha", "user": "alpha"}}
    )
    data["current-context"] = "beta"
    stat = path.stat()
    path.write_text(yaml.safe_dump(data))
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    new = k8s_config._load_config_for_context("alpha")
    assert new is not old
    assert new.configuration.host == "https://new.invalid"
    assert bearer_token(new.configuration) == "Bearer synthetic-new-token"
    assert k8s_config.get_active_context() == "beta"
    assert (
        k8s_config._load_config_for_context().configuration.host
        == "https://new.invalid"
    )


@pytest.mark.parametrize("change", ["modify", "delete", "replace"])
def test_watcher_uses_primary_pathlist_and_invalidates_clients(
    tmp_path, monkeypatch, change
):
    a = write_config(tmp_path / "a")
    b = write_config(tmp_path / "b", name="beta", current=None)
    ignored = write_config(tmp_path / "ignored", name="ignored", current="ignored")
    monkeypatch.setenv("MCP_K8S_KUBECONFIG", os.pathsep.join([str(a), str(b)]))
    monkeypatch.setenv("KUBECONFIG", str(ignored))
    monkeypatch.setattr(k8s_config.KubeconfigWatcher, "start", lambda self: None)
    k8s_config.enable_kubeconfig_watch()
    watcher = k8s_config._kubeconfig_watcher
    assert set(watcher._watched_files) == {str(a), str(b)}
    original = k8s_config._load_config_for_context("alpha")
    notifications = []

    def changed():
        # Callback re-entry must not run under the watcher's file lock.
        watcher.add_file(str(a))
        notifications.append(True)

    k8s_config.on_config_change(changed)
    if change == "delete":
        b.unlink()
    elif change == "replace":
        stat = b.stat()
        replacement = write_config(tmp_path / "replacement", name="beta", current=None)
        os.utime(replacement, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        replacement.replace(b)
    else:
        write_config(b, name="beta", server="https://changed.invalid", current=None)
        stat = b.stat()
        os.utime(b, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    worker = threading.Thread(target=watcher._check_files, daemon=True)
    worker.start()
    worker.join(timeout=3)
    assert not worker.is_alive(), "watcher callback deadlocked"
    assert notifications == [True]
    if change == "delete":
        with pytest.raises(ProviderError):
            k8s_config._load_config_for_context("alpha")
        write_config(b, name="beta", current=None)
        watcher._check_files()
        assert notifications == [True, True]
    assert k8s_config._load_config_for_context("alpha") is not original


def test_watcher_tracks_creation_of_missing_config(tmp_path):
    path = tmp_path / "later"
    watcher = k8s_config.KubeconfigWatcher()
    watcher.add_file(str(path))
    notifications = []
    k8s_config.on_config_change(lambda: notifications.append(True))
    write_config(path)
    watcher._check_files()
    assert notifications == [True]


@pytest.mark.parametrize("stateless", [False, True])
@pytest.mark.parametrize(
    "provider_type, allowed", [("single", "alpha"), ("in-cluster", "in-cluster")]
)
def test_explicit_context_stays_within_provider_boundary(
    tmp_path, isolated_config, monkeypatch, stateless, provider_type, allowed
):
    _, in_cluster = isolated_config
    path = write_config(tmp_path / "config", current=None)
    monkeypatch.setenv("MCP_K8S_KUBECONFIG", str(path))
    monkeypatch.setenv("MCP_K8S_PROVIDER", provider_type)
    monkeypatch.setenv("MCP_K8S_CONTEXT", "alpha")

    def synthetic_incluster(client_configuration=None, **kwargs):
        target = client_configuration or client.Configuration()
        target.host = "https://in-cluster.invalid"
        if client_configuration is None:
            client.Configuration.set_default(target)

    in_cluster.side_effect = synthetic_incluster
    k8s_config.set_stateless_mode(stateless)
    assert k8s_config.get_active_context() == allowed
    assert [ctx["name"] for ctx in k8s_config.list_contexts()] == [allowed]
    expected_host = (
        "https://alpha.invalid"
        if provider_type == "single"
        else "https://in-cluster.invalid"
    )
    assert (
        k8s_config._load_config_for_context(allowed).configuration.host == expected_host
    )
    assert k8s_config._load_config_for_context().configuration.host == expected_host
    with pytest.raises(UnknownContextError) as error:
        k8s_config._load_config_for_context("other")
    assert error.value.available == [allowed]


def test_unknown_context_is_distinct_from_configuration_failure(tmp_path):
    provider = KubernetesProvider(
        ProviderConfig(kubeconfig_path=str(write_config(tmp_path / "config")))
    )
    with pytest.raises(UnknownContextError) as error:
        provider.get_api_client("missing")
    assert error.value.available == ["alpha"]


def test_cache_clear_is_serialized_with_client_creation(tmp_path, monkeypatch):
    provider = KubernetesProvider(
        ProviderConfig(kubeconfig_path=str(write_config(tmp_path / "config")))
    )
    started = threading.Event()
    release = threading.Event()
    original_init = client.ApiClient.__init__

    def blocked_init(self, *args, **kwargs):
        started.set()
        assert release.wait(timeout=3)
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(client.ApiClient, "__init__", blocked_init)
    with ThreadPoolExecutor(max_workers=2) as executor:
        loading = executor.submit(provider.get_api_client, "alpha")
        assert started.wait(timeout=3)
        clearing = executor.submit(provider.clear_client_cache)
        release.set()
        original = loading.result(timeout=3)
        clearing.result(timeout=3)
    monkeypatch.setattr(client.ApiClient, "__init__", original_init)
    assert provider.get_api_client("alpha") is not original


def test_implicit_default_missing_can_use_incluster(isolated_config):
    _, in_cluster = isolated_config
    in_cluster.side_effect = None
    provider = KubernetesProvider()
    assert provider.get_current_context() == "in-cluster"
    with pytest.raises(UnknownContextError):
        provider.get_api_client("alpha")


def test_explicit_context_request_cannot_activate_implicit_incluster(isolated_config):
    _, in_cluster = isolated_config
    in_cluster.side_effect = None
    with pytest.raises((ProviderError, UnknownContextError)):
        k8s_config._load_config_for_context("alpha")


def test_single_provider_requires_context(tmp_path):
    with pytest.raises(ProviderError, match="MCP_K8S_CONTEXT"):
        KubernetesProvider(
            ProviderConfig(
                provider_type=ProviderType.SINGLE,
                kubeconfig_path=str(write_config(tmp_path / "config")),
            )
        )


def test_kubectl_env_copies_environment_and_expands_primary_pathlist(
    isolated_config, monkeypatch
):
    synthetic_home, in_cluster = isolated_config
    monkeypatch.setenv("MCP_K8S_KUBECONFIG", os.pathsep.join(["~/one", "", "~/two"]))
    monkeypatch.setenv("KUBECONFIG", "~/ignored")
    original = dict(os.environ)
    environment = k8s_config.get_kubectl_env()
    assert environment["KUBECONFIG"] == os.pathsep.join(
        [str(synthetic_home / "one"), str(synthetic_home / "two")]
    )
    assert dict(os.environ) == original
    assert environment is not os.environ
    assert environment["MCP_K8S_KUBECONFIG"] == original["MCP_K8S_KUBECONFIG"]
    in_cluster.assert_not_called()


@pytest.mark.parametrize("environment_path", ["~/from-env", ""])
def test_kubectl_env_uses_environment_or_default(
    isolated_config, monkeypatch, environment_path
):
    synthetic_home, _ = isolated_config
    monkeypatch.setenv("MCP_K8S_KUBECONFIG", "")
    monkeypatch.setenv("KUBECONFIG", environment_path)
    original = dict(os.environ)
    expected = synthetic_home / ("from-env" if environment_path else ".kube/config")
    assert k8s_config.get_kubectl_env()["KUBECONFIG"] == str(expected)
    assert dict(os.environ) == original


def test_kubectl_env_uses_existing_provider_source(tmp_path, monkeypatch):
    primary = write_config(tmp_path / "primary")
    provider = KubernetesProvider(ProviderConfig(kubeconfig_path=str(primary)))
    monkeypatch.setattr(KubernetesProvider, "_instance", provider)
    monkeypatch.setenv("MCP_K8S_KUBECONFIG", str(tmp_path / "other"))
    assert k8s_config.get_kubectl_env()["KUBECONFIG"] == provider.config.kubeconfig_path


def test_valid_empty_configuration_is_not_a_read_failure(tmp_path, monkeypatch):
    path = tmp_path / "empty-config"
    path.write_text(
        yaml.safe_dump(
            {
                "apiVersion": "v1",
                "kind": "Config",
                "contexts": [],
                "clusters": [],
                "users": [],
            }
        )
    )
    monkeypatch.setenv("MCP_K8S_KUBECONFIG", str(path))
    assert k8s_config.list_contexts() == []
    assert k8s_config.get_active_context() is None
    with pytest.raises(UnknownContextError) as error:
        k8s_config._load_config_for_context("alpha")
    assert error.value.available == []
    with pytest.raises(ProviderError, match="[Cc]ontext"):
        k8s_config._load_config_for_context()


def test_secondary_file_error_is_not_ignored(tmp_path, monkeypatch):
    first = write_config(tmp_path / "first", current=None)
    second = tmp_path / "second"
    second.write_text("clusters: [")
    monkeypatch.setenv("MCP_K8S_KUBECONFIG", os.pathsep.join([str(first), str(second)]))
    with pytest.raises(ProviderError, match="second"):
        k8s_config._load_config_for_context("alpha")


def test_unreadable_configuration_propagates_original_error(tmp_path, monkeypatch):
    path = write_config(tmp_path / "config")
    monkeypatch.setenv("MCP_K8S_KUBECONFIG", str(path))
    real_open = builtins.open

    def unreadable(filename, *args, **kwargs):
        if str(filename) == str(path):
            raise PermissionError("synthetic permission denied")
        return real_open(filename, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", unreadable)
    with pytest.raises(ProviderError, match="Permission denied") as error:
        k8s_config.list_contexts()
    assert isinstance(error.value.__cause__, PermissionError)


@pytest.mark.parametrize("stateless", [False, True])
def test_context_override_is_shared_by_default_clients_and_list(
    tmp_path, monkeypatch, stateless
):
    path = write_config(tmp_path / "config")
    data = yaml.safe_load(path.read_text())
    data["contexts"].append(
        {"name": "beta", "context": {"cluster": "alpha", "user": "alpha"}}
    )
    path.write_text(yaml.safe_dump(data))
    monkeypatch.setenv("MCP_K8S_KUBECONFIG", str(path))
    monkeypatch.setenv("MCP_K8S_CONTEXT", "beta")
    k8s_config.set_stateless_mode(stateless)
    assert k8s_config.get_active_context() == "beta"
    assert [ctx["name"] for ctx in k8s_config.list_contexts() if ctx["active"]] == [
        "beta"
    ]
    assert (
        k8s_config._load_config_for_context().configuration.host
        == "https://alpha.invalid"
    )


def test_incluster_client_does_not_reuse_foreign_sdk_default(
    tmp_path, monkeypatch, isolated_config
):
    _, in_cluster = isolated_config
    foreign = client.Configuration()
    foreign.host = "https://foreign.invalid"
    client.Configuration.set_default(foreign)

    def synthetic_incluster(client_configuration=None):
        assert client_configuration is not None
        client_configuration.host = "https://service-account.invalid"

    in_cluster.side_effect = synthetic_incluster
    provider = KubernetesProvider(ProviderConfig(provider_type=ProviderType.IN_CLUSTER))
    assert (
        provider.get_api_client("in-cluster").configuration.host
        == "https://service-account.invalid"
    )
    assert client.Configuration.get_default_copy().host == "https://foreign.invalid"


@pytest.mark.parametrize("server", [None, ""])
def test_cluster_without_server_is_configuration_error(tmp_path, monkeypatch, server):
    path = write_config(tmp_path / "config")
    data = yaml.safe_load(path.read_text())
    if server is None:
        del data["clusters"][0]["cluster"]["server"]
    else:
        data["clusters"][0]["cluster"]["server"] = server
    path.write_text(yaml.safe_dump(data))
    monkeypatch.setenv("MCP_K8S_KUBECONFIG", str(path))
    with pytest.raises(ProviderError, match="[Ss]erver"):
        k8s_config._load_config_for_context("alpha")


@pytest.mark.parametrize("stateless", [False, True])
@pytest.mark.parametrize("provider_type", ["kubeconfig", "single"])
@pytest.mark.parametrize("retired", ["cluster", "user", "server", "current"])
def test_explicit_context_ignores_unrelated_retired_configuration(
    tmp_path, monkeypatch, stateless, retired, provider_type
):
    path = write_config(tmp_path / "config")
    data = yaml.safe_load(path.read_text())
    if retired == "current":
        data["current-context"] = "retired"
    else:
        values = {"cluster": "alpha", "user": "alpha"}
        if retired == "cluster":
            values["cluster"] = "deleted"
        elif retired == "user":
            values["user"] = "deleted"
        else:
            values["cluster"] = "deleted-server"
            data["clusters"].append({"name": "deleted-server", "cluster": {}})
        data["contexts"].append({"name": "retired", "context": values})
    path.write_text(yaml.safe_dump(data))
    monkeypatch.setenv("MCP_K8S_KUBECONFIG", str(path))
    monkeypatch.setenv("MCP_K8S_PROVIDER", provider_type)
    if provider_type == "single":
        monkeypatch.setenv("MCP_K8S_CONTEXT", "alpha")
    k8s_config.set_stateless_mode(stateless)
    expected = (
        ["alpha"]
        if retired == "current" or provider_type == "single"
        else ["alpha", "retired"]
    )
    assert [ctx["name"] for ctx in k8s_config.list_contexts()] == expected
    assert (
        k8s_config._load_config_for_context("alpha").configuration.host
        == "https://alpha.invalid"
    )
    if provider_type == "single":
        assert k8s_config.get_active_context() == "alpha"
        assert (
            k8s_config._load_config_for_context().configuration.host
            == "https://alpha.invalid"
        )
        with pytest.raises(UnknownContextError):
            k8s_config._load_config_for_context("retired")
    elif retired == "current":
        assert k8s_config.get_active_context() == "retired"
        with pytest.raises(ProviderError, match="[Cc]ontext"):
            k8s_config._load_config_for_context()
    else:
        with pytest.raises(ProviderError):
            k8s_config._load_config_for_context("retired")


def test_yaml_extension_values_keep_configuration_fingerprint_stable(tmp_path):
    path = write_config(tmp_path / "config")
    data = yaml.safe_load(path.read_text())
    data["extensions"] = [
        {
            "name": "history",
            "extension": {
                "last-used": datetime(2026, 11, 1, tzinfo=timezone.utc),
                "date": date(2026, 11, 1),
                "tags": {"alpha", "beta"},
                "payload": b"synthetic",
            },
        }
    ]
    path.write_text(yaml.safe_dump(data))
    provider = KubernetesProvider(ProviderConfig(kubeconfig_path=str(path)))
    original = provider.get_api_client("alpha")
    assert original.configuration.host == "https://alpha.invalid"
    assert provider.get_api_client("alpha") is original


@pytest.mark.asyncio
async def test_sdk_credential_errors_do_not_leak_into_tool_output(
    tmp_path, monkeypatch, caplog
):
    from fastmcp import Client, FastMCP
    from kubectl_mcp_tool.tools.cluster import register_cluster_tools

    path = write_config(tmp_path / "config")
    marker = "SYNTHETIC_CREDENTIAL_REVIEW_MARKER"
    data = yaml.safe_load(path.read_text())
    data["users"][0]["user"] = {
        "auth-provider": {
            "name": "oidc",
            "config": {"client-secret": marker},
        }
    }
    path.write_text(yaml.safe_dump(data))
    monkeypatch.setenv("MCP_K8S_KUBECONFIG", str(path))
    server = FastMCP("synthetic-credential-errors")
    register_cluster_tools(server, non_destructive=True)
    async with Client(server) as session:
        result = await session.session.call_tool(
            "get_nodes_summary", {"context": "alpha"}
        )
    assert result.structuredContent["success"] is False
    assert marker not in json.dumps(result.structuredContent)
    assert marker not in str(result.content)
    assert marker not in caplog.text
    assert "id-token" in result.structuredContent["error"]
    with pytest.raises(ProviderError) as error:
        k8s_config._load_config_for_context("alpha")
    assert marker not in str(error.value)
    assert marker not in str(error.value.__cause__)
    assert marker not in "".join(traceback.format_exception(error.value))


@pytest.mark.parametrize("persist", [False, True])
@pytest.mark.parametrize("positional", [False, True])
def test_sdk_refresh_honors_persistence_and_loader_flags(
    tmp_path, monkeypatch, persist, positional
):
    from kubernetes.config.kube_config import KubeConfigLoader

    path = write_config(tmp_path / "config")
    data = yaml.safe_load(path.read_text())
    data["users"][0]["user"] = {
        "auth-provider": {
            "name": "gcp",
            "config": {
                "access-token": "synthetic-expired",
                "expiry": "2000-01-01T00:00:00Z",
            },
        }
    }
    path.write_text(yaml.safe_dump(data))
    monkeypatch.setenv("MCP_K8S_KUBECONFIG", str(path))
    temp_files = str(tmp_path / "sdk-temp")
    refreshes = []
    loader_flags = []
    original_init = KubeConfigLoader.__init__

    def synthetic_init(self, *args, **kwargs):
        def refresh():
            refreshes.append(True)
            return SimpleNamespace(
                token="synthetic-fresh",
                expiry=datetime.now(timezone.utc) + timedelta(hours=1),
            )

        loader_flags.append(kwargs.get("temp_file_path"))
        kwargs["get_google_credentials"] = refresh
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(KubeConfigLoader, "__init__", synthetic_init)
    for _ in range(2):
        target = client.Configuration()
        if positional:
            config.load_kube_config(None, "alpha", target, persist, temp_files)
        else:
            config.load_kube_config(
                context="alpha",
                client_configuration=target,
                persist_config=persist,
                temp_file_path=temp_files,
            )
        assert bearer_token(target) == "Bearer synthetic-fresh"
    stored = yaml.safe_load(path.read_text())["users"][0]["user"]["auth-provider"][
        "config"
    ]
    assert stored["access-token"] == (
        "synthetic-fresh" if persist else "synthetic-expired"
    )
    assert len(refreshes) == (1 if persist else 2)
    assert loader_flags == [temp_files, temp_files]


def test_refresh_persists_only_winning_user_source_without_rewriting_relative_paths(
    tmp_path, monkeypatch
):
    from kubernetes.config.kube_config import KubeConfigLoader

    clusters = write_config(tmp_path / "clusters" / "config")
    data = yaml.safe_load(clusters.read_text())
    data["users"] = []
    clusters.write_text(yaml.safe_dump(data))
    users = tmp_path / "users" / "config"
    users.parent.mkdir()
    user = {
        "name": "alpha",
        "user": {
            "auth-provider": {
                "name": "gcp",
                "config": {
                    "access-token": "synthetic-expired",
                    "expiry": "2000-01-01T00:00:00Z",
                },
            },
            "client-certificate": "client.pem",
            "client-key": "client.key",
        },
    }
    users.write_text(yaml.safe_dump({"users": [user]}))
    (users.parent / "client.pem").write_text("synthetic-cert")
    (users.parent / "client.key").write_text("synthetic-key")
    shadow = write_config(tmp_path / "shadow", current=None)
    unchanged = {clusters: clusters.read_bytes(), shadow: shadow.read_bytes()}
    refreshes = []
    original_init = KubeConfigLoader.__init__

    def synthetic_init(self, *args, **kwargs):
        def refresh():
            refreshes.append(True)
            return SimpleNamespace(
                token="synthetic-fresh",
                expiry=datetime.now(timezone.utc) + timedelta(hours=1),
            )

        kwargs["get_google_credentials"] = refresh
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(KubeConfigLoader, "__init__", synthetic_init)
    provider = KubernetesProvider(
        ProviderConfig(
            kubeconfig_path=os.pathsep.join(map(str, [clusters, users, shadow]))
        )
    )
    assert bearer_token(provider.get_configuration("alpha")) == "Bearer synthetic-fresh"
    stored = yaml.safe_load(users.read_text())["users"][0]["user"]
    assert stored["auth-provider"]["config"]["access-token"] == "synthetic-fresh"
    assert stored["client-certificate"] == "client.pem"
    assert stored["client-key"] == "client.key"
    assert bearer_token(provider.get_configuration("alpha")) == "Bearer synthetic-fresh"
    assert refreshes == [True]
    for path, original in unchanged.items():
        assert path.read_bytes() == original


@pytest.mark.parametrize("persist", [False, True])
def test_oidc_refresh_token_writeback_respects_persistence(
    tmp_path, monkeypatch, persist
):
    from kubernetes.config.kube_config import KubeConfigLoader

    def jwt(expiry):
        payload = (
            base64.urlsafe_b64encode(json.dumps({"exp": expiry}).encode())
            .decode()
            .rstrip("=")
        )
        return f"e30.{payload}.c3ludGhldGlj"

    expired, fresh = jwt(1), jwt(4102444800)
    path = write_config(tmp_path / "config")
    data = yaml.safe_load(path.read_text())
    data["users"][0]["user"] = {
        "auth-provider": {
            "name": "oidc",
            "config": {
                "id-token": expired,
                "refresh-token": "synthetic-expired-refresh",
            },
        }
    }
    path.write_text(yaml.safe_dump(data))
    monkeypatch.setenv("MCP_K8S_KUBECONFIG", str(path))
    refreshed = []

    def offline_refresh(self, provider):
        refreshed.append(True)
        provider["config"].value["id-token"] = fresh
        provider["config"].value["refresh-token"] = "synthetic-fresh-refresh"

    monkeypatch.setattr(KubeConfigLoader, "_refresh_oidc", offline_refresh)
    for _ in range(2):
        target = client.Configuration()
        config.load_kube_config(
            context="alpha", client_configuration=target, persist_config=persist
        )
        assert bearer_token(target) == f"Bearer {fresh}"
    stored = yaml.safe_load(path.read_text())["users"][0]["user"]["auth-provider"][
        "config"
    ]
    assert stored["id-token"] == (fresh if persist else expired)
    assert stored["refresh-token"] == (
        "synthetic-fresh-refresh" if persist else "synthetic-expired-refresh"
    )
    assert len(refreshed) == (1 if persist else 2)


@pytest.fixture
def retained_refresh(monkeypatch):
    """Force the real SDK refresh path while replacing only offline credentials."""
    from kubernetes.config.kube_config import KubeConfigLoader

    refreshed = []
    original_init = KubeConfigLoader.__init__

    def offline_init(self, *args, **kwargs):
        def credentials():
            refreshed.append(True)
            return SimpleNamespace(
                token=f"synthetic-refreshed-{len(refreshed)}",
                expiry=datetime.now(timezone.utc) + timedelta(hours=1),
            )

        kwargs["get_google_credentials"] = credentials
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(KubeConfigLoader, "__init__", offline_init)

    def force(configuration):
        before = len(refreshed)
        with monkeypatch.context() as expired:
            expired.setattr(kube_config, "_is_expired", lambda *args: True)
            configuration.refresh_api_key_hook(configuration)
        assert len(refreshed) == before + 1, "SDK credential refresh was not triggered"

    return force, refreshed


def write_refresh_config(path):
    write_config(path, server="https://old.invalid")
    data = yaml.safe_load(path.read_text())
    data["users"][0]["user"] = {
        "auth-provider": {
            "name": "gcp",
            "config": {
                "access-token": "synthetic-initial",
                "expiry": "2099-01-01T00:00:00Z",
            },
        }
    }
    path.write_text(yaml.safe_dump(data))
    return data


@pytest.mark.parametrize("provider_reread", [False, True])
@pytest.mark.parametrize("change", ["endpoint", "metadata", "credentials", "comment"])
def test_retained_sdk_refresh_cannot_overwrite_changed_source(
    tmp_path, retained_refresh, provider_reread, change
):
    force, refreshed = retained_refresh
    path = tmp_path / "config"
    data = write_refresh_config(path)
    provider = KubernetesProvider(ProviderConfig(kubeconfig_path=str(path)))
    retained = provider.get_api_client("alpha")
    assert refreshed == []
    if change == "endpoint":
        data["clusters"][0]["cluster"]["server"] = "https://new.invalid"
        data["extensions"] = [{"name": "new-metadata", "extension": {"generation": 2}}]
    elif change == "metadata":
        data["extensions"] = [{"name": "new-metadata", "extension": {"generation": 2}}]
    elif change == "credentials":
        data["users"][0]["user"] = {
            "auth-provider": {
                "name": "gcp",
                "config": {
                    "access-token": "synthetic-external-rotation",
                    "expiry": "2099-01-01T00:00:00Z",
                },
            },
            "fixture-owner": "replacement",
        }
    contents = yaml.safe_dump(data)
    if change == "comment":
        contents += "# external generation changed\n"
    path.write_text(contents)
    expected = path.read_bytes()
    if provider_reread:
        provider.get_api_client("alpha")
    force(retained.configuration)
    assert refreshed == [True]
    assert path.read_bytes() == expected
    current = provider.get_api_client("alpha").configuration
    assert current.host == data["clusters"][0]["cluster"]["server"]
    expected_token = data["users"][0]["user"]["auth-provider"]["config"]["access-token"]
    assert bearer_token(current) == f"Bearer {expected_token}"


@pytest.mark.parametrize("provider_reread", [False, True])
def test_retained_sdk_refresh_cannot_restore_replaced_winning_user_source(
    tmp_path, retained_refresh, provider_reread
):
    force, refreshed = retained_refresh
    clusters = tmp_path / "clusters"
    data = write_refresh_config(clusters)
    original_user = data["users"][0]
    data["users"] = []
    clusters.write_text(yaml.safe_dump(data))
    first = tmp_path / "first-user"
    first.write_text(yaml.safe_dump({"users": [original_user]}))
    replacement = tmp_path / "replacement-user"
    new_user = deepcopy(original_user)
    new_user["user"]["auth-provider"]["config"][
        "access-token"
    ] = "synthetic-replacement"
    replacement.write_text(yaml.safe_dump({"users": [new_user]}))
    provider = KubernetesProvider(
        ProviderConfig(
            kubeconfig_path=os.pathsep.join(map(str, [clusters, first, replacement]))
        )
    )
    retained = provider.get_api_client("alpha")
    original_user["name"] = "retired"
    first.write_text(yaml.safe_dump({"users": [original_user]}))
    expected = {path: path.read_bytes() for path in (clusters, first, replacement)}
    if provider_reread:
        assert (
            bearer_token(provider.get_api_client("alpha").configuration)
            == "Bearer synthetic-replacement"
        )
    force(retained.configuration)
    assert refreshed == [True]
    for path, contents in expected.items():
        assert path.read_bytes() == contents
    assert (
        bearer_token(provider.get_api_client("alpha").configuration)
        == "Bearer synthetic-replacement"
    )


@pytest.mark.parametrize("provider_reread", [False, True])
def test_retained_sdk_refresh_advances_its_source_generation_after_each_write(
    tmp_path, retained_refresh, provider_reread
):
    force, refreshed = retained_refresh
    path = tmp_path / "config"
    data = write_refresh_config(path)
    data["extensions"] = [{"name": "metadata", "extension": {"generation": 1}}]
    path.write_text(yaml.safe_dump(data))
    provider = KubernetesProvider(ProviderConfig(kubeconfig_path=str(path)))
    retained = provider.get_api_client("alpha")
    for count in (1, 2):
        force(retained.configuration)
        stored = yaml.safe_load(path.read_text())
        assert (
            stored["users"][0]["user"]["auth-provider"]["config"]["access-token"]
            == f"synthetic-refreshed-{count}"
        )
        assert stored["clusters"] == data["clusters"]
        assert stored["contexts"] == data["contexts"]
        assert stored["extensions"] == data["extensions"]
        if provider_reread:
            provider.get_api_client("alpha")
    assert refreshed == [True, True]
