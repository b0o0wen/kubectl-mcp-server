"""Provider tests prepared for the local 1.25 backport, not upstream 2.0."""

import os
import socket
from unittest.mock import Mock

import pytest
import yaml
from kubernetes import client, config
from kubernetes.config import kube_config
from kubernetes.config.config_exception import ConfigException

pytestmark = pytest.mark.unit


def _write_kubeconfig(path, context_name, current=True):
    """Write an actual synthetic kubeconfig, optionally without a default."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "apiVersion": "v1",
        "kind": "Config",
        "clusters": [
            {
                "name": f"{context_name}-cluster",
                "cluster": {"server": f"https://{context_name}.invalid"},
            }
        ],
        "users": [
            {
                "name": f"{context_name}-user",
                "user": {"token": f"synthetic-{context_name}-token"},
            }
        ],
        "contexts": [
            {
                "name": context_name,
                "context": {
                    "cluster": f"{context_name}-cluster",
                    "user": f"{context_name}-user",
                },
            }
        ],
    }
    if current:
        data["current-context"] = context_name
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


def _bearer_token(configuration):
    # SDK 37 renamed the storage key; SDK 28/35 use "authorization".
    return configuration.api_key.get(
        "authorization", configuration.api_key.get("BearerToken")
    )


@pytest.fixture(autouse=True)
def synthetic_configuration(tmp_path, monkeypatch):
    """Keep config loading real while isolating defaults, identities, and traffic."""
    synthetic_home = tmp_path / "synthetic-home"
    default_path = synthetic_home / ".kube" / "config"
    _write_kubeconfig(default_path, "fixture")
    real_expanduser = os.path.expanduser

    def expanduser(path):
        if str(path).startswith("~/"):
            return str(synthetic_home / str(path)[2:])
        return real_expanduser(path)

    monkeypatch.setattr(os.path, "expanduser", expanduser)
    monkeypatch.setattr(kube_config, "KUBE_CONFIG_DEFAULT_LOCATION", str(default_path))
    for key in (
        "MCP_K8S_PROVIDER",
        "MCP_K8S_KUBECONFIG",
        "MCP_K8S_CONTEXT",
        "KUBECONFIG",
        "MCP_STATELESS_MODE",
        "MCP_KUBECONFIG_WATCH",
        "KUBERNETES_SERVICE_HOST",
        "KUBERNETES_SERVICE_PORT",
    ):
        monkeypatch.delenv(key, raising=False)
    # Set a synthetic source before importing the editable package: legacy
    # startup code must also stay away from the host's real kubeconfig.
    monkeypatch.setenv("KUBECONFIG", str(default_path))
    in_cluster = Mock(side_effect=AssertionError("unexpected in-cluster identity"))
    monkeypatch.setattr(config, "load_incluster_config", in_cluster)
    monkeypatch.setattr(
        socket.socket, "connect", Mock(side_effect=AssertionError("network forbidden"))
    )
    monkeypatch.setattr(
        client.ApiClient,
        "call_api",
        Mock(side_effect=AssertionError("cluster traffic forbidden")),
    )
    opened_clients = []
    original_init = client.ApiClient.__init__

    def tracked_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        opened_clients.append(self)

    monkeypatch.setattr(client.ApiClient, "__init__", tracked_init)
    from kubectl_mcp_tool import k8s_config
    from kubectl_mcp_tool.providers import KubernetesProvider

    monkeypatch.setattr(KubernetesProvider, "_instance", None)
    monkeypatch.setattr(k8s_config, "_config_loaded", False)
    monkeypatch.setattr(k8s_config, "_stateless_mode", False)
    monkeypatch.setattr(client.Configuration, "_default", None)
    yield synthetic_home, in_cluster
    for api_client in opened_clients:
        api_client.close()


class TestProviderConfig:
    def test_expands_each_path_in_kubeconfig_path_list(
        self, synthetic_configuration, monkeypatch
    ):
        from kubectl_mcp_tool.providers import ProviderConfig

        synthetic_home, _ = synthetic_configuration
        monkeypatch.setenv(
            "KUBECONFIG", os.pathsep.join(["~/.kube/sit", "~/.kube/stg"])
        )
        original_environment = dict(os.environ)
        provider_config = ProviderConfig.from_env()

        assert provider_config.kubeconfig_path == os.pathsep.join(
            [str(synthetic_home / ".kube" / "sit"), str(synthetic_home / ".kube" / "stg")]
        )
        assert dict(os.environ) == original_environment


class TestKubernetesProvider:
    def test_loads_contexts_from_multiple_kubeconfig_files(
        self, tmp_path, synthetic_configuration
    ):
        from kubectl_mcp_tool.providers import KubernetesProvider, ProviderConfig

        _, in_cluster = synthetic_configuration
        first_path = tmp_path / "sit.config"
        second_path = tmp_path / "stg.config"
        _write_kubeconfig(first_path, "sit", current=False)
        _write_kubeconfig(second_path, "stg", current=False)
        provider = KubernetesProvider(
            ProviderConfig(
                kubeconfig_path=os.pathsep.join([str(first_path), str(second_path)])
            )
        )

        assert [context.name for context in provider.list_contexts()] == ["sit", "stg"]
        assert provider.get_current_context() is None
        for context_name in ("sit", "stg"):
            loaded = provider.get_api_client(context_name).configuration
            assert loaded.host == f"https://{context_name}.invalid"
            assert _bearer_token(loaded) == f"Bearer synthetic-{context_name}-token"
        in_cluster.assert_not_called()

    @pytest.mark.parametrize("multiple_files", [False, True])
    def test_kubernetes_client_merges_contexts_from_path_list(
        self, tmp_path, multiple_files
    ):
        """Keep the SDK's single-file and multi-file context behavior unchanged."""
        first_path = tmp_path / "sit.config"
        _write_kubeconfig(first_path, "sit")
        paths = [str(first_path)]
        expected_names = ["sit"]
        if multiple_files:
            second_path = tmp_path / "stg.config"
            _write_kubeconfig(second_path, "stg")
            paths.append(str(second_path))
            expected_names.append("stg")
        path_list = os.pathsep.join(paths)

        contexts, active = config.list_kube_config_contexts(config_file=path_list)
        assert [context["name"] for context in contexts] == expected_names
        # The SDK's current-context merge priority differs between versions;
        # this check intentionally preserves the legacy SDK compatibility rule.
        assert active["name"] in expected_names
        for context_name in expected_names:
            target = client.Configuration()
            config.load_kube_config(
                config_file=path_list,
                context=context_name,
                client_configuration=target,
                persist_config=False,
            )
            assert target.host == f"https://{context_name}.invalid"
            assert _bearer_token(target) == f"Bearer synthetic-{context_name}-token"


class TestProviderFailureBoundaries:
    @pytest.mark.parametrize("stateless", [False, True])
    def test_authentication_error_never_falls_back_to_another_identity(
        self, tmp_path, monkeypatch, synthetic_configuration, stateless
    ):
        from kubectl_mcp_tool import k8s_config
        from kubectl_mcp_tool.providers import ProviderError

        _, in_cluster = synthetic_configuration
        primary = tmp_path / "sit.config"
        alternate = tmp_path / "stg.config"
        _write_kubeconfig(primary, "sit")
        _write_kubeconfig(alternate, "stg")
        missing_token = tmp_path / "missing-token"
        data = yaml.safe_load(primary.read_text())
        data["users"][0]["user"] = {"tokenFile": str(missing_token)}
        primary.write_text(yaml.safe_dump(data), encoding="utf-8")
        monkeypatch.setenv("MCP_K8S_KUBECONFIG", str(primary))
        monkeypatch.setenv("KUBECONFIG", str(alternate))
        # Both a viable fallback loader and a foreign SDK default are present.
        # The primary config's authentication failure must still propagate.
        in_cluster.side_effect = None
        foreign = client.Configuration()
        foreign.host = "https://foreign-identity.invalid"
        client.Configuration.set_default(foreign)
        k8s_config.set_stateless_mode(stateless)

        with pytest.raises(ProviderError) as error:
            k8s_config._load_config_for_context("sit")
        # SDK 35 raises FileNotFoundError directly for tokenFile; older SDKs
        # may wrap the same missing-file failure in ConfigException.
        assert isinstance(error.value.__cause__, (FileNotFoundError, ConfigException))
        assert str(missing_token) in str(error.value)
        in_cluster.assert_not_called()
        assert client.Configuration.get_default_copy().host == foreign.host

    @pytest.mark.parametrize("one_file_exists", [False, True])
    def test_explicit_missing_path_is_provider_error_without_identity_fallback(
        self, tmp_path, monkeypatch, synthetic_configuration, one_file_exists
    ):
        from kubectl_mcp_tool import k8s_config
        from kubectl_mcp_tool.providers import ProviderError

        _, in_cluster = synthetic_configuration
        first_path = tmp_path / "sit.config"
        missing_path = tmp_path / "missing-stg.config"
        if one_file_exists:
            _write_kubeconfig(first_path, "sit")
        path_list = os.pathsep.join([str(first_path), str(missing_path)])
        monkeypatch.setenv("MCP_K8S_KUBECONFIG", path_list)
        in_cluster.side_effect = None

        with pytest.raises(ProviderError) as error:
            k8s_config._load_config_for_context("sit")
        assert isinstance(error.value.__cause__, FileNotFoundError)
        assert path_list in str(error.value)
        in_cluster.assert_not_called()
