"""
Kubernetes configuration loader utility.

Handles both in-cluster and out-of-cluster (kubeconfig) configurations.
Supports multi-cluster operations with context targeting.

This module provides context-aware client creation for multi-cluster support.
All get_*_client() functions accept an optional 'context' parameter.

Environment Variables:
    MCP_K8S_PROVIDER: Provider type (kubeconfig, in-cluster, single)
    MCP_K8S_KUBECONFIG: Path to kubeconfig file
    MCP_K8S_CONTEXT: Default context for single provider
    MCP_K8S_QPS: API rate limit (default: 100)
    MCP_K8S_BURST: API burst limit (default: 200)
    MCP_K8S_TIMEOUT: Request timeout in seconds (default: 30)
    MCP_STATELESS_MODE: If "true", don't cache API clients (reload config each request)
    MCP_KUBECONFIG_WATCH: If "true", auto-detect kubeconfig file changes
"""

import os
import inspect
import logging
import threading
from copy import deepcopy
from functools import wraps
from typing import Optional, Any, List, Dict, Callable

logger = logging.getLogger("mcp-server")

_stateless_mode = os.environ.get("MCP_STATELESS_MODE", "").lower() in (
    "true",
    "1",
    "yes",
)
_kubeconfig_watcher: Optional["KubeconfigWatcher"] = None
_config_change_callbacks: List[Callable[[], None]] = []


class KubeconfigWatcher:
    """Watch kubeconfig files for changes and trigger reloads.

    This class monitors kubeconfig files and automatically invalidates
    cached configurations when changes are detected.
    """

    def __init__(self, check_interval: float = 5.0):
        """Initialize the kubeconfig watcher.

        Args:
            check_interval: How often to check for changes (seconds)
        """
        self._check_interval = check_interval
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._watched_files: Dict[str, Any] = {}
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

    def start(self):
        """Start watching for kubeconfig changes."""
        if self._running:
            return

        self._stop_event.clear()
        self._running = True
        self._thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._thread.start()
        logger.info("Kubeconfig watcher started")

    def stop(self):
        """Stop watching for kubeconfig changes."""
        self._running = False
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        logger.info("Kubeconfig watcher stopped")

    def add_file(self, filepath: str):
        """Add a file to watch.

        Args:
            filepath: Path to kubeconfig file
        """
        filepath = os.path.expanduser(filepath)
        with self._lock:
            self._watched_files[filepath] = self._file_signature(filepath)

    @staticmethod
    def _file_signature(filepath):
        try:
            stat = os.stat(filepath)
        except FileNotFoundError:
            return None
        return (
            stat.st_mtime_ns,
            stat.st_ctime_ns,
            stat.st_size,
            stat.st_ino,
            stat.st_dev,
        )

    def _watch_loop(self):
        while not self._stop_event.is_set():
            try:
                self._check_files()
            except Exception as error:
                logger.debug(f"Kubeconfig watch error: {error}")
            self._stop_event.wait(self._check_interval)

    def _check_files(self):
        changed = False
        with self._lock:
            for filepath, previous in list(self._watched_files.items()):
                current = self._file_signature(filepath)
                if current != previous:
                    self._watched_files[filepath] = current
                    changed = True
        # Callbacks can add watched files or load clients: never hold the file
        # lock while calling provider code or user callbacks.
        if changed:
            self._on_config_changed()

    def _on_config_changed(self):
        global _config_loaded
        _config_loaded = False
        # A watcher notification must not initialize a different provider.
        with KubernetesProvider._instance_lock:
            provider = KubernetesProvider._instance
        if provider is not None:
            provider.invalidate_cache()
        for callback in list(_config_change_callbacks):
            try:
                callback()
            except Exception as error:
                logger.debug(f"Config change callback error: {error}")


def enable_kubeconfig_watch(check_interval: float = 5.0):
    """Enable automatic kubeconfig file watching.

    When enabled, the server will automatically detect changes to kubeconfig
    files and reload the configuration. This is useful when:
    - Cloud provider CLIs update credentials (aws, gcloud, az)
    - Users switch contexts using external tools
    - Kubeconfig files are mounted dynamically (e.g., in Kubernetes)

    Args:
        check_interval: How often to check for changes (seconds)

    Example:
        enable_kubeconfig_watch(check_interval=10.0)
    """
    global _kubeconfig_watcher

    if _kubeconfig_watcher is not None:
        return

    watcher = KubeconfigWatcher(check_interval=check_interval)
    with KubernetesProvider._instance_lock:
        provider = KubernetesProvider._instance
        provider_config = (
            provider.config if provider is not None else ProviderConfig.from_env()
        )
    if provider_config.provider_type != ProviderType.IN_CLUSTER:
        for path in get_kubeconfig_paths(provider_config.kubeconfig_path):
            watcher.add_file(path)
    _kubeconfig_watcher = watcher
    watcher.start()


def disable_kubeconfig_watch():
    """Disable kubeconfig file watching."""
    global _kubeconfig_watcher

    if _kubeconfig_watcher is not None:
        _kubeconfig_watcher.stop()
        _kubeconfig_watcher = None


def on_config_change(callback: Callable[[], None]):
    """Register a callback to be called when kubeconfig changes.

    Args:
        callback: Function to call when config changes
    """
    _config_change_callbacks.append(callback)


def is_stateless_mode() -> bool:
    """Check if stateless mode is enabled.

    In stateless mode, API clients are not cached and configuration
    is reloaded on each request. This is useful for:
    - Serverless environments (Lambda, Cloud Functions)
    - Environments where credentials may change frequently
    - Security-conscious deployments

    Returns:
        True if stateless mode is enabled
    """
    return _stateless_mode


def set_stateless_mode(enabled: bool):
    """Enable or disable stateless mode.

    Args:
        enabled: True to enable stateless mode
    """
    global _stateless_mode
    _stateless_mode = enabled
    if enabled:
        logger.info("Stateless mode enabled - API clients will not be cached")
    else:
        logger.info("Stateless mode disabled - API clients will be cached")


from .providers import (
    KubernetesProvider,
    ProviderConfig,
    ProviderType,
    ProviderError,
    UnknownContextError,
    get_provider,
    get_context_names,
    get_current_context as provider_get_current_context,
    get_kubeconfig_paths,
    resolve_kubeconfig_source,
    validate_context,
)

_HAS_PROVIDER = True
_config_loaded = False
_original_load_kube_config = None


def load_kubernetes_config(
    context: str = "",
    client_configuration=None,
    persist_config=True,
    temp_file_path=None,
):
    """Load the authoritative provider source, propagating configuration errors.

    An explicit target Configuration stays isolated. Bare startup calls set
    the SDK default for compatibility with legacy clients.
    """
    global _config_loaded
    from kubernetes import client

    configuration = get_provider().get_configuration(
        context, persist_config=persist_config, temp_file_path=temp_file_path
    )
    if client_configuration is None:
        client.Configuration.set_default(configuration)
    else:
        client_configuration.__dict__.update(deepcopy(configuration.__dict__))
    _config_loaded = True
    return True


def _patched_load_kube_config(*args, **kwargs):
    """Honor explicit SDK files; route default-source loads through the provider."""
    global _config_loaded
    # Bind the actual installed SDK signature so positional arguments,
    # defaults, duplicate arguments and unsupported flags retain SDK semantics.
    bound = inspect.signature(_original_load_kube_config).bind(*args, **kwargs)
    bound.apply_defaults()
    parameters = bound.arguments
    if parameters.get("config_file") is not None:
        result = _original_load_kube_config(*args, **kwargs)
        _config_loaded = True
        return result
    load_kubernetes_config(
        parameters.get("context") or "",
        client_configuration=parameters.get("client_configuration"),
        persist_config=parameters.get("persist_config", True),
        temp_file_path=parameters.get("temp_file_path"),
    )


def patch_kubernetes_config():
    """Route legacy default-source SDK loads through the same provider."""
    global _original_load_kube_config
    from kubernetes import config

    if _original_load_kube_config is None:
        _original_load_kube_config = config.load_kube_config
        config.load_kube_config = wraps(_original_load_kube_config)(
            _patched_load_kube_config
        )


patch_kubernetes_config()


def _load_config_for_context(context: str = "") -> Any:
    """Create a context client from one source, optionally bypassing its cache."""
    return get_provider().get_api_client(context, use_cache=not _stateless_mode)


def get_kubectl_env() -> Dict[str, str]:
    """Return a subprocess environment using the SDK provider's pathlist.

    This does not load credentials, read kubeconfig files, initialize a
    provider, or mutate the process environment.
    """
    with KubernetesProvider._instance_lock:
        provider = KubernetesProvider._instance
        pathlist = (
            provider.config.kubeconfig_path
            if provider is not None
            else resolve_kubeconfig_source()[0]
        )
    environment = os.environ.copy()
    environment["KUBECONFIG"] = os.pathsep.join(get_kubeconfig_paths(pathlist))
    return environment


def _get_client(context: str, client_class):
    """Helper to create a configured Kubernetes API client."""
    try:
        api_client = _load_config_for_context(context)
        return client_class(api_client=api_client)
    except (ProviderError, UnknownContextError):
        raise
    except Exception as e:
        raise RuntimeError(
            f"Invalid kube-config. Context: {context or 'default'}. Error: {e}"
        ) from e


def get_k8s_client(context: str = ""):
    """Get a configured Kubernetes Core API client."""
    from kubernetes import client

    return _get_client(context, client.CoreV1Api)


def get_apps_client(context: str = ""):
    """Get a configured Kubernetes Apps API client."""
    from kubernetes import client

    return _get_client(context, client.AppsV1Api)


def get_rbac_client(context: str = ""):
    """Get a configured Kubernetes RBAC API client."""
    from kubernetes import client

    return _get_client(context, client.RbacAuthorizationV1Api)


def get_networking_client(context: str = ""):
    """Get a configured Kubernetes Networking API client."""
    from kubernetes import client

    return _get_client(context, client.NetworkingV1Api)


def get_storage_client(context: str = ""):
    """Get a configured Kubernetes Storage API client."""
    from kubernetes import client

    return _get_client(context, client.StorageV1Api)


def get_batch_client(context: str = ""):
    """Get a configured Kubernetes Batch API client."""
    from kubernetes import client

    return _get_client(context, client.BatchV1Api)


def get_autoscaling_client(context: str = ""):
    """Get a configured Kubernetes Autoscaling API client."""
    from kubernetes import client

    return _get_client(context, client.AutoscalingV1Api)


def get_policy_client(context: str = ""):
    """Get a configured Kubernetes Policy API client."""
    from kubernetes import client

    return _get_client(context, client.PolicyV1Api)


def get_custom_objects_client(context: str = ""):
    """Get a configured Kubernetes Custom Objects API client."""
    from kubernetes import client

    return _get_client(context, client.CustomObjectsApi)


def get_version_client(context: str = ""):
    """Get a configured Kubernetes Version API client."""
    from kubernetes import client

    return _get_client(context, client.VersionApi)


def get_admissionregistration_client(context: str = ""):
    """Get a configured Kubernetes Admission Registration API client."""
    from kubernetes import client

    return _get_client(context, client.AdmissionregistrationV1Api)


def get_apiextensions_client(context: str = ""):
    """Get a configured Kubernetes API Extensions client."""
    from kubernetes import client

    return _get_client(context, client.ApiextensionsV1Api)


def get_coordination_client(context: str = ""):
    """Get a configured Kubernetes Coordination API client."""
    from kubernetes import client

    return _get_client(context, client.CoordinationV1Api)


def get_events_client(context: str = ""):
    """Get a configured Kubernetes Events API client."""
    from kubernetes import client

    return _get_client(context, client.EventsV1Api)


def list_contexts() -> list:
    """List contexts without concealing a failed configuration read."""
    return [
        {
            "name": ctx.name,
            "cluster": ctx.cluster,
            "user": ctx.user,
            "namespace": ctx.namespace,
            "active": ctx.is_active,
        }
        for ctx in get_provider().list_contexts()
    ]


def get_active_context() -> Optional[str]:
    """Return the configured active context, or None when none is selected."""
    return provider_get_current_context()


def context_exists(context: str) -> bool:
    """Check if a context exists in kubeconfig."""
    contexts = list_contexts()
    return any(ctx["name"] == context for ctx in contexts)


def _get_kubectl_context_args(context: str = "") -> list:
    """Get kubectl command arguments for specifying a context."""
    if context and context.strip():
        return ["--context", context.strip()]
    return []


__all__ = [
    "get_k8s_client",
    "get_apps_client",
    "get_rbac_client",
    "get_networking_client",
    "get_storage_client",
    "get_batch_client",
    "get_autoscaling_client",
    "get_policy_client",
    "get_custom_objects_client",
    "get_version_client",
    "get_admissionregistration_client",
    "get_apiextensions_client",
    "get_coordination_client",
    "get_events_client",
    "load_kubernetes_config",
    "patch_kubernetes_config",
    "list_contexts",
    "get_active_context",
    "context_exists",
    "enable_kubeconfig_watch",
    "disable_kubeconfig_watch",
    "on_config_change",
    "KubeconfigWatcher",
    "is_stateless_mode",
    "set_stateless_mode",
    "get_kubeconfig_paths",
    "resolve_kubeconfig_source",
    "get_kubectl_env",
    "KubernetesProvider",
    "ProviderConfig",
    "ProviderType",
    "ProviderError",
    "UnknownContextError",
    "get_provider",
    "get_context_names",
    "validate_context",
]
