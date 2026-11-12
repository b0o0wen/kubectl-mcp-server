import hashlib
import logging
import os
import re
import threading
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import yaml

logger = logging.getLogger("mcp-server")


class ProviderType(Enum):
    """Kubernetes provider types."""

    KUBECONFIG = "kubeconfig"
    IN_CLUSTER = "in-cluster"
    SINGLE = "single"


class UnknownContextError(Exception):
    """Raised when a requested context is not found in a valid configuration."""

    def __init__(self, context: str, available: Optional[List[str]] = None):
        self.context = context
        self.available = available or []
        msg = f"Context '{context}' not found"
        if self.available:
            msg += f". Available contexts: {', '.join(self.available)}"
        super().__init__(msg)


class ProviderError(Exception):
    """Raised when provider configuration is invalid or cannot be read."""


def _raise_configuration_error(message: str, error: Exception):
    """Expose a bounded diagnostic and a safe cause, never SDK node contents."""
    from kubernetes.config.config_exception import ConfigException

    if isinstance(error, yaml.YAMLError):
        detail = "Invalid YAML"
        mark = getattr(error, "problem_mark", None)
        if mark is not None:
            detail += f" at line {mark.line + 1}, column {mark.column + 1}"
    elif isinstance(error, PermissionError):
        detail = "Permission denied"
    elif isinstance(error, FileNotFoundError):
        detail = "File not found"
        if error.filename:
            detail += f": {error.filename}"
    elif isinstance(error, ConfigException):
        detail = "Invalid kubeconfig"
        missing = re.search(r"Expected key ([a-zA-Z0-9_-]+)\b", str(error))
        if missing and missing.group(1) in {
            "server",
            "cluster",
            "clusters",
            "contexts",
            "users",
            "user",
            "name",
            "current-context",
            "id-token",
            "access-token",
            "refresh-token",
            "client-id",
            "client-secret",
            "idp-issuer-url",
        }:
            detail += f": missing or invalid field '{missing.group(1)}'"
    else:
        detail = f"Configuration failure ({type(error).__name__})"
    try:
        cause = type(error)(detail)
    except Exception:
        cause = Exception(detail)
    # Keep the traceback locations while replacing error text and YAML marks,
    # which may carry source lines or credentials in their repr/str methods.
    cause = cause.with_traceback(error.__traceback__)
    raise ProviderError(f"{message}: {detail}") from cause


def resolve_kubeconfig_source() -> Tuple[str, bool]:
    """Return the authoritative pathlist and whether it was explicitly set.

    This is a pure source lookup: it never reads files or changes environment
    variables. Subprocess callers can use the returned pathlist in their own
    KUBECONFIG environment without changing the server's identity.
    """
    for key in ("MCP_K8S_KUBECONFIG", "KUBECONFIG"):
        value = os.environ.get(key)
        if value:
            return os.pathsep.join(get_kubeconfig_paths(value)), True
    return os.pathsep.join(get_kubeconfig_paths("~/.kube/config")), False


def get_kubeconfig_paths(pathlist: Optional[str] = None) -> List[str]:
    """Split a pathlist, expand each path, and retain first-occurrence order."""
    if pathlist is None:
        pathlist, _ = resolve_kubeconfig_source()
    return list(
        dict.fromkeys(
            os.path.expanduser(path) for path in pathlist.split(os.pathsep) if path
        )
    )


@dataclass
class ProviderConfig:
    """Configuration for Kubernetes provider."""

    provider_type: ProviderType = ProviderType.KUBECONFIG
    kubeconfig_path: str = ""
    context: str = ""
    qps: float = 100.0
    burst: int = 200
    timeout: int = 30
    kubeconfig_explicit: Optional[bool] = None

    def __post_init__(self):
        if not self.kubeconfig_path:
            self.kubeconfig_path, explicit = resolve_kubeconfig_source()
        else:
            self.kubeconfig_path = os.pathsep.join(
                get_kubeconfig_paths(self.kubeconfig_path)
            )
            explicit = True
        if self.kubeconfig_explicit is None:
            self.kubeconfig_explicit = explicit

    @classmethod
    def from_env(cls) -> "ProviderConfig":
        """Create config from environment variables."""
        provider_str = os.environ.get("MCP_K8S_PROVIDER", "kubeconfig").lower()
        try:
            provider_type = ProviderType(provider_str)
        except ValueError:
            logger.warning(f"Unknown provider type '{provider_str}', using kubeconfig")
            provider_type = ProviderType.KUBECONFIG
        pathlist, explicit = resolve_kubeconfig_source()
        return cls(
            provider_type=provider_type,
            kubeconfig_path=pathlist,
            kubeconfig_explicit=explicit,
            context=os.environ.get("MCP_K8S_CONTEXT", ""),
            qps=float(os.environ.get("MCP_K8S_QPS", "100")),
            burst=int(os.environ.get("MCP_K8S_BURST", "200")),
            timeout=int(os.environ.get("MCP_K8S_TIMEOUT", "30")),
        )


@dataclass
class ContextInfo:
    """Information about a kubeconfig context."""

    name: str
    cluster: str
    user: str
    namespace: str = "default"
    is_active: bool = False


def _config_files_fingerprint(config_files):
    return hashlib.sha256(
        yaml.safe_dump(config_files, sort_keys=True).encode("utf-8")
    ).digest()


def _read_kubeconfig(pathlist: str):
    """Read an SDK merger without requiring or selecting current-context.

    The SDK keeps first-wins named entries and their source ConfigNode paths.
    It silently skips missing files and overwrites current-context, so enforce
    complete source reads and first-wins current-context here.
    """
    from kubernetes.config.config_exception import ConfigException
    from kubernetes.config.kube_config import KubeConfigMerger

    try:
        paths = get_kubeconfig_paths(pathlist)
        if not paths:
            raise ConfigException("No kubeconfig paths configured")
        for path in paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Kubeconfig not found: {path}")
        merger = KubeConfigMerger(os.pathsep.join(paths))
        if merger.config is None:
            raise ConfigException("No Kubernetes configuration found")
        active = None
        for path in paths:
            if path not in merger.config_files:
                raise FileNotFoundError(f"Kubeconfig not read: {path}")
            data = merger.config_files[path]
            if not isinstance(data, dict):
                raise ConfigException(f"Expected a kubeconfig mapping in {path}")
            current = data.get("current-context")
            if current is not None and not isinstance(current, str):
                raise ConfigException(
                    f"Expected current-context to be a string in {path}"
                )
            if active is None and current:
                active = current
            for section, body in (
                ("clusters", "cluster"),
                ("users", "user"),
                ("contexts", "context"),
            ):
                entries = data.get(section) or []
                if not isinstance(entries, list):
                    raise ConfigException(f"Expected {section} to be a list in {path}")
                names = set()
                for entry in entries:
                    if (
                        not isinstance(entry, dict)
                        or not isinstance(entry.get("name"), str)
                        or not entry["name"]
                        or not isinstance(entry.get(body), dict)
                    ):
                        raise ConfigException(f"Invalid {section} entry in {path}")
                    if entry["name"] in names:
                        raise ConfigException(
                            f"Duplicate {section} name '{entry['name']}' in {path}"
                        )
                    names.add(entry["name"])
        merger.config.value["current-context"] = active
        for section in ("clusters", "users", "contexts"):
            for index, node in enumerate(merger.config.value[section]):
                # KubeConfigMerger embeds the entire entry dict in node.name.
                # SDK errors inherit that name, including auth-provider secrets.
                node.name = f"{node.path}/{section}[{index}]"
        # Pair the SDK's parsed snapshot with exact source bytes. A callback
        # must not adopt a newer generation merely because this provider hasn't
        # reloaded yet, or because only comments/formatting changed.
        observed_files = {}
        generation = []
        for path in paths:
            with open(path, "rb") as source:
                contents = source.read()
            observed_files[path] = yaml.safe_load(contents)
            generation.append((path, hashlib.sha256(contents).hexdigest()))
        if _config_files_fingerprint(observed_files) != _config_files_fingerprint(
            merger.config_files
        ):
            raise ProviderError(
                f"Kubeconfig '{pathlist}' changed while being read; retry the request"
            )
        merger.source_generation = tuple(generation)
        fingerprint = hashlib.sha256(
            yaml.safe_dump(generation).encode("utf-8")
        ).digest()
        return merger, active, fingerprint
    except ProviderError:
        raise
    except Exception as error:
        _raise_configuration_error(f"Failed to read kubeconfig '{pathlist}'", error)


class KubernetesProvider:
    """Manage clients for one authoritative configuration source."""

    _instance: Optional["KubernetesProvider"] = None
    _instance_lock = threading.RLock()

    def __init__(self, config: Optional[ProviderConfig] = None):
        self.config = config or ProviderConfig.from_env()
        self._lock = threading.RLock()
        self._api_clients: Dict[str, Any] = {}
        self._in_cluster = False
        self._contexts_cache: Optional[List[ContextInfo]] = None
        self._active_context: Optional[str] = None
        self._merger = None
        self._fingerprint = None
        self._initialize()

    @classmethod
    def get_instance(cls) -> "KubernetesProvider":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @classmethod
    def reset_instance(cls):
        """Reset the singleton without disrupting clients already in use."""
        with cls._instance_lock:
            cls._instance = None

    def _initialize(self):
        if self.config.provider_type == ProviderType.IN_CLUSTER:
            self._initialize_in_cluster()
            return
        if self.config.provider_type == ProviderType.SINGLE and not self.config.context:
            raise ProviderError("MCP_K8S_CONTEXT must be set for 'single' provider")
        paths = get_kubeconfig_paths(self.config.kubeconfig_path)
        if (
            self.config.provider_type == ProviderType.KUBECONFIG
            and not self.config.kubeconfig_explicit
            and not self.config.context
            and len(paths) == 1
            and not os.path.exists(paths[0])
        ):
            self._initialize_in_cluster()
            return
        self._refresh_contexts_cache()

    def _initialize_in_cluster(self):
        from kubernetes import client, config

        try:
            # Never use a process-global default left by another identity.
            config.load_incluster_config(client_configuration=client.Configuration())
        except Exception as error:
            _raise_configuration_error("Failed to load in-cluster config", error)
        self._in_cluster = True
        self._active_context = "in-cluster"

    def _refresh_contexts_cache(self):
        if self._in_cluster:
            return
        with self._lock:
            try:
                merger, current, fingerprint = _read_kubeconfig(
                    self.config.kubeconfig_path
                )
                active = self.config.context or current
                contexts = []
                for node in merger.config.value["contexts"]:
                    data = node.value
                    values = data["context"]
                    contexts.append(
                        ContextInfo(
                            name=data["name"],
                            cluster=values.get("cluster") or "",
                            user=values.get("user") or "",
                            namespace=values.get("namespace") or "default",
                            is_active=data["name"] == active,
                        )
                    )
                if fingerprint != self._fingerprint:
                    self._api_clients.clear()
                self._merger = merger
                self._fingerprint = fingerprint
                self._active_context = active
                self._contexts_cache = contexts
            except Exception:
                self.invalidate_cache()
                raise

    def list_contexts(self) -> List[ContextInfo]:
        with self._lock:
            if self._in_cluster:
                return [
                    ContextInfo(
                        "in-cluster", "in-cluster", "service-account", is_active=True
                    )
                ]
            self._refresh_contexts_cache()
            contexts = self._contexts_cache or []
            if self.config.provider_type == ProviderType.SINGLE:
                contexts = [ctx for ctx in contexts if ctx.name == self.config.context]
            return list(contexts)

    def get_current_context(self) -> Optional[str]:
        with self._lock:
            self._refresh_contexts_cache()
            return self._active_context

    def validate_context(self, context: str) -> str:
        with self._lock:
            self._refresh_contexts_cache()
            return self._resolve_context(context)

    def _resolve_context(self, context: str) -> str:
        if self._in_cluster:
            available = ["in-cluster"]
        elif self.config.provider_type == ProviderType.SINGLE:
            available = [self.config.context]
        else:
            available = [ctx.name for ctx in self._contexts_cache or []]
        resolved = context or self._active_context
        if not resolved:
            raise ProviderError(
                "No current context configured; specify a context explicitly"
            )
        if resolved not in available:
            if not context:
                raise ProviderError(
                    f"Current context '{resolved}' not found; specify a valid context explicitly"
                )
            raise UnknownContextError(resolved, available)
        return resolved

    def _create_configuration(
        self, context: str, persist_config=True, temp_file_path=None
    ):
        from kubernetes import client, config
        from kubernetes.config.config_exception import ConfigException
        from kubernetes.config.kube_config import KubeConfigLoader

        try:
            configuration = client.Configuration()
            if self._in_cluster:
                config.load_incluster_config(client_configuration=configuration)
            else:
                merger = self._merger
                # Keep the original file dictionaries unchanged. Only the
                # selected loader's copy gets absolute credential paths.
                loader_config = deepcopy(merger.config)
                values = loader_config["contexts"].get_with_name(context)["context"]
                cluster_name = values.safe_get("cluster")
                user_name = values.safe_get("user")
                clusters = {
                    node.value["name"]: node.value["cluster"]
                    for node in loader_config.value["clusters"]
                }
                users = {node.value["name"] for node in loader_config.value["users"]}
                if not isinstance(cluster_name, str) or cluster_name not in clusters:
                    raise ProviderError(
                        f"Invalid cluster for selected context '{context}'"
                    )
                server = clusters[cluster_name].get("server")
                if not isinstance(server, str) or not server:
                    raise ProviderError(
                        f"Invalid server for selected context '{context}'"
                    )
                if user_name and (
                    not isinstance(user_name, str) or user_name not in users
                ):
                    raise ProviderError(
                        f"Invalid user for selected context '{context}'"
                    )
                cluster_node = loader_config["clusters"].get_with_name(cluster_name)
                user_node = (
                    loader_config["users"].get_with_name(user_name)
                    if user_name
                    else None
                )
                for node, body, fields in (
                    (cluster_node, "cluster", ("certificate-authority",)),
                    (
                        user_node,
                        "user",
                        ("client-certificate", "client-key", "tokenFile"),
                    ),
                ):
                    if node is None:
                        continue
                    for field in fields:
                        value = node.value[body].get(field)
                        if value:
                            if not isinstance(value, str):
                                raise ConfigException(
                                    f"Invalid kube-config file. Expected key {field}"
                                )
                            node.value[body][field] = os.path.abspath(
                                os.path.join(
                                    os.path.dirname(node.path),
                                    os.path.expanduser(value),
                                )
                            )

                captured_paths = tuple(merger.paths)
                expected_generation = merger.source_generation

                def persist_refresh():
                    nonlocal expected_generation
                    # Retained SDK clients can refresh after the provider has
                    # reloaded. Check disk content, not the current self._merger.
                    with self._lock:
                        try:
                            if (
                                user_node is None
                                or "auth-provider" not in user_node.value["user"]
                            ):
                                return
                            if (
                                tuple(get_kubeconfig_paths(self.config.kubeconfig_path))
                                != captured_paths
                            ):
                                return
                            current, _, _ = _read_kubeconfig(
                                os.pathsep.join(captured_paths)
                            )
                            if current.source_generation != expected_generation:
                                return
                            current_user = current.config["users"].get_with_name(
                                user_name, safe=True
                            )
                            if (
                                current_user is None
                                or current_user.path != user_node.path
                            ):
                                return
                            # Update only authentication in the freshly read
                            # winning source, never a whole captured merger.
                            current_user.value["user"]["auth-provider"] = deepcopy(
                                user_node.value["user"]["auth-provider"]
                            )
                            contents = yaml.safe_dump(
                                current.config_files[current_user.path],
                                default_flow_style=False,
                            ).encode("utf-8")
                            with open(current_user.path, "r+b") as target:
                                actual = hashlib.sha256(target.read()).hexdigest()
                                for path, expected in expected_generation:
                                    if path == current_user.path:
                                        digest = actual
                                    else:
                                        with open(path, "rb") as source:
                                            digest = hashlib.sha256(
                                                source.read()
                                            ).hexdigest()
                                    if digest != expected:
                                        return
                                target.seek(0)
                                target.write(contents)
                                target.truncate()
                            # Derive the next generation from exactly our write.
                            # Don't accidentally adopt a subsequent external edit.
                            written = hashlib.sha256(contents).hexdigest()
                            expected_generation = tuple(
                                (path, written if path == current_user.path else digest)
                                for path, digest in expected_generation
                            )
                        except ProviderError:
                            raise
                        except Exception as error:
                            _raise_configuration_error(
                                "Failed to persist refreshed kubeconfig", error
                            )

                loader = KubeConfigLoader(
                    config_dict=loader_config,
                    active_context=context,
                    config_base_path=None,
                    config_persister=persist_refresh if persist_config else None,
                    temp_file_path=temp_file_path,
                )
                loader.load_and_set(configuration)
            return configuration
        except ProviderError:
            self.invalidate_cache()
            raise
        except Exception as error:
            self.invalidate_cache()
            _raise_configuration_error(
                f"Failed to load context '{context}' from '{self.config.kubeconfig_path}'",
                error,
            )

    def get_configuration(
        self, context: str = "", persist_config=True, temp_file_path=None
    ):
        """Load an isolated Configuration with the SDK's persistence policy."""
        with self._lock:
            self._refresh_contexts_cache()
            return self._create_configuration(
                self._resolve_context(context),
                persist_config=persist_config,
                temp_file_path=temp_file_path,
            )

    def get_api_client(self, context: str = "", use_cache: bool = True) -> Any:
        from kubernetes import client

        with self._lock:
            self._refresh_contexts_cache()
            resolved = self._resolve_context(context)
            if use_cache and resolved in self._api_clients:
                return self._api_clients[resolved]
            api_client = client.ApiClient(
                configuration=self._create_configuration(resolved)
            )
            if use_cache:
                self._api_clients[resolved] = api_client
            return api_client

    def clear_client_cache(self, context: str = ""):
        """Serialize invalidation with client creation; don't close in-use clients."""
        with self._lock:
            if context:
                self._api_clients.pop(context, None)
            else:
                self._api_clients.clear()

    def invalidate_cache(self):
        with self._lock:
            self._api_clients.clear()
            self._contexts_cache = None
            self._merger = None
            self._fingerprint = None
            if not self._in_cluster:
                self._active_context = None


def get_provider() -> KubernetesProvider:
    return KubernetesProvider.get_instance()


def get_context_names() -> List[str]:
    return [ctx.name for ctx in get_provider().list_contexts()]


def get_current_context() -> Optional[str]:
    return get_provider().get_current_context()


def validate_context(context: str) -> str:
    return get_provider().validate_context(context)
