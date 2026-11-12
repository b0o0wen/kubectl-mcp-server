"""Protocol regressions for declared success/error output schemas, without clusters."""

import json
import subprocess
from unittest.mock import patch

import jsonschema
import pytest
from fastmcp import Client, FastMCP

from kubectl_mcp_tool import schemas, structured
from kubectl_mcp_tool.providers import ProviderError
from kubectl_mcp_tool.tools.cluster import register_cluster_tools
from kubectl_mcp_tool.tools.cost import register_cost_tools
from kubectl_mcp_tool.tools.helm import register_helm_tools
from kubectl_mcp_tool.tools.kubectl import register_kubectl_tools
from kubectl_mcp_tool.tools.operations import register_operations_tools
from kubectl_mcp_tool.tools.pods import register_pod_tools
from kubectl_mcp_tool.tools.security import register_security_tools


@pytest.fixture
def server():
    server = FastMCP("context-error-contract")
    register_cluster_tools(server, non_destructive=True)
    register_pod_tools(server, non_destructive=True)
    register_security_tools(server, non_destructive=True)
    register_cost_tools(server, non_destructive=True)
    register_helm_tools(server, non_destructive=True, check_helm_fn=lambda: True)
    register_kubectl_tools(server, non_destructive=True)
    register_operations_tools(server, non_destructive=True)
    return server


async def call_protocol(server, name, arguments):
    # Use the SDK session to exercise tools/call and output-schema validation.
    # Calling the Python function alone would miss the original regression.
    async with Client(server) as client:
        tools = {tool.name: tool for tool in await client.list_tools()}
        result = await client.session.call_tool(name, arguments)
    assert not result.isError, result.content
    assert result.structuredContent is not None
    schema = tools[name].outputSchema
    assert schema["type"] == "object"
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(result.structuredContent, schema)
    assert json.loads(result.content[0].text) == result.structuredContent
    return result.structuredContent


@pytest.mark.asyncio
@pytest.mark.parametrize("contexts", [[], [{"name": "qa", "is_active": False}]])
async def test_list_contexts_without_default_returns_null_active_context(server, contexts):
    with patch("kubectl_mcp_tool.tools.cluster._list_contexts_impl", return_value=contexts), patch(
        "kubectl_mcp_tool.tools.cluster.get_active_context", return_value=None
    ):
        result = await call_protocol(server, "list_contexts", {})
    assert result == {
        "success": True,
        "contexts": contexts,
        "active_context": None,
        "total": len(contexts),
    }


DECLARED_READ_TOOLS = [
    ("list_contexts", {}, "cluster._list_contexts_impl"),
    ("get_nodes_summary", {"context": "qa"}, "cluster.get_k8s_client"),
    ("check_pod_health", {"pod_name": "web", "context": "qa"}, "pods.get_k8s_client"),
    ("analyze_network_policies", {"context": "qa"}, "security.get_networking_client"),
    ("get_cost_analysis", {"context": "qa"}, "cost.get_k8s_client"),
    ("helm_list", {"context": "qa"}, "helm.subprocess.run"),
    ("helm_status", {"release_name": "web", "context": "qa"}, "helm.subprocess.run"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("name,arguments,boundary", DECLARED_READ_TOOLS, ids=[
    case[0] for case in DECLARED_READ_TOOLS
])
async def test_declared_read_tool_preserves_original_error(server, name, arguments, boundary):
    message = "Invalid kubeconfig: context 'qa' has no matching cluster"
    with patch("kubectl_mcp_tool.tools." + boundary, side_effect=ProviderError(message)):
        result = await call_protocol(server, name, arguments)
    assert result == {"success": False, "error": message}


# Include the stateful read models retained by 1.25, even though their tools
# were removed from this 2.0 checkout. Exercise their contracts over MCP too.
RESPONSE_MODELS = [
    schemas.GetPodsResponse,
    schemas.GetLogsResponse,
    schemas.GetPodEventsResponse,
    schemas.CheckPodHealthResponse,
    schemas.GetDeploymentsResponse,
    schemas.DeploymentHistoryResponse,
    schemas.CheckDeploymentHealthResponse,
    schemas.GetNamespacesResponse,
    schemas.GetConfigMapsResponse,
    schemas.GetServicesResponse,
    schemas.GetNodesSummaryResponse,
    schemas.ListContextsResponse,
    schemas.GetIngressResponse,
    schemas.GetStorageClassesResponse,
    schemas.GetPVCsResponse,
    schemas.AnalyzeNetworkPoliciesResponse,
    schemas.GetRBACRolesResponse,
    schemas.HelmListResponse,
    schemas.HelmStatusResponse,
    schemas.GetCostAnalysisResponse,
]


@pytest.mark.asyncio
@pytest.mark.parametrize("model", RESPONSE_MODELS, ids=lambda model: model.__name__)
async def test_all_response_contracts_preserve_error_metadata_over_protocol(model):
    server = FastMCP("synthetic-read-error")
    name = {
        schemas.GetPodsResponse: "get_pods",
        schemas.GetDeploymentsResponse: "get_deployments",
    }.get(model, "read_resource")
    error = {
        "success": False,
        "error": "Context 'missing' not found. Available contexts: qa",
        "context": "missing",
        "hint": "Specify an available context",
        "details": {"available_contexts": ["qa"]},
    }

    @server.tool(name=name, output_schema=structured.response_schema(model))
    def read_resource() -> dict:
        return structured.structured_response(error, model)

    assert await call_protocol(server, name, {}) == error


@pytest.mark.asyncio
@pytest.mark.parametrize("model,field,item,expected_item", [
    (schemas.GetPodsResponse, "pods",
     {"name": "web", "namespace": "default", "status": "Running"},
     {"name": "web", "namespace": "default", "status": "Running", "ip": None}),
    (schemas.GetDeploymentsResponse, "deployments",
     {"name": "web", "namespace": "default", "ready": 1, "available": 1},
     {"name": "web", "namespace": "default", "ready": 1, "available": 1, "replicas": None}),
])
async def test_success_shape_and_optional_defaults_are_preserved(model, field, item, expected_item):
    server = FastMCP("synthetic-read-success")

    @server.tool(output_schema=structured.response_schema(model))
    def read_resource() -> dict:
        return structured.structured_response({
            "success": True, "context": "qa", field: [item], "extra": "strip me"
        }, model)

    assert await call_protocol(server, "read_resource", {}) == {
        "success": True, "context": "qa", field: [expected_item]
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("model", RESPONSE_MODELS, ids=lambda model: model.__name__)
async def test_advertised_contract_rejects_incomplete_success_and_malformed_error(model):
    server = FastMCP("strict-output-contract")

    @server.tool(output_schema=structured.response_schema(model))
    def read_resource() -> dict:
        return {"success": False, "error": "unavailable"}

    async with Client(server) as client:
        tool = next(tool for tool in await client.list_tools() if tool.name == "read_resource")
    validator = jsonschema.Draft202012Validator(tool.outputSchema)
    for malformed in (
        {"success": True},
        {"success": True, "error": "cannot mask incomplete success"},
        {"success": False},
        {"success": False, "error": None},
        {"success": False, "error": 42},
        {"success": "false", "error": "unavailable"},
    ):
        with pytest.raises(jsonschema.ValidationError):
            validator.validate(malformed)


@pytest.mark.asyncio
@pytest.mark.parametrize("name,arguments", [
    ("helm_list", {}),
    ("helm_status", {"release_name": "web"}),
])
async def test_helm_unavailable_early_error_is_schema_valid(name, arguments):
    server = FastMCP("helm-unavailable")
    register_helm_tools(server, non_destructive=True, check_helm_fn=lambda: False)
    assert await call_protocol(server, name, arguments) == {
        "success": False, "error": "Helm is not available on this system"
    }


SUBPROCESS_READS = [
    ("kubectl_get", {"resource_type": "pods", "context": "qa"}, "utils", "[]"),
    ("kubectl_generic", {"command": "get pods", "context": "qa"}, "operations", "[]"),
    ("kubectl_describe", {"resource_type": "pod", "name": "web", "context": "qa"},
     "operations", "pod details"),
    ("backup_resource", {"resource_type": "pod", "name": "web", "context": "qa"},
     "operations", "kind: Pod"),
    ("wait_for_condition", {"resource_type": "pod", "name": "web", "condition": "condition=Ready",
                            "context": "qa"}, "operations", "condition met"),
    ("kubeconfig_view", {"minify": False}, "cluster", '{"contexts": []}'),
    ("get_api_resources", {"context": "qa"}, "cluster", "NAME APIVERSION"),
    ("kubectl_explain", {"resource": "pods", "context": "qa"}, "cluster", "KIND: Pod"),
    ("node_logs", {"name": "worker", "context": "qa"}, "cluster", "kubelet started"),
    ("helm_list", {"context": "qa"}, "helm", "[]"),
    ("helm_status", {"release_name": "web", "context": "qa"}, "helm", "{}"),
    ("helm_history", {"release_name": "web", "context": "qa"}, "helm", "[]"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("returncode", [0, 1], ids=["success", "failure"])
@pytest.mark.parametrize("name,arguments,module,stdout", SUBPROCESS_READS, ids=[
    case[0] for case in SUBPROCESS_READS
])
async def test_subprocess_reads_use_canonical_kubeconfig_environment(
    server, name, arguments, module, stdout, returncode
):
    # Core tests cover normalization and environment copying. This boundary
    # regression verifies each consumer passes the shared helper's exact env.
    environment = {"KUBECONFIG": "/synthetic/first.yaml:/synthetic/second.yaml", "PATH": "/bin"}
    failure = "Context 'qa' has no matching cluster"
    completed = subprocess.CompletedProcess(
        [], returncode, stdout=stdout if returncode == 0 else "", stderr=failure if returncode else ""
    )
    with patch("kubectl_mcp_tool.tools." + module + ".get_kubectl_env",
               return_value=environment, create=True), patch(
        "kubectl_mcp_tool.tools." + module + ".subprocess.run", return_value=completed
    ) as run:
        result = await call_protocol(server, name, arguments)
    assert result["success"] is (returncode == 0)
    if returncode:
        assert result.get("error", result.get("output")) == failure
    assert run.call_args.kwargs.get("env") == environment
    command = run.call_args.args[0]
    if arguments.get("context"):
        flag = "--kube-context" if command[0] == "helm" else "--context"
        assert command[command.index(flag) + 1] == "qa"


HELM_WRITES = [
    ("install_helm_chart", set(), True),
    ("install_helm_chart", {"get_namespace"}, True),
    ("install_helm_chart", {"get_namespace", "create_namespace"}, False),
    ("install_helm_chart", {"install"}, False),
    ("upgrade_helm_chart", set(), True),
    ("upgrade_helm_chart", {"upgrade"}, False),
    ("uninstall_helm_chart", set(), True),
    ("uninstall_helm_chart", {"uninstall"}, False),
    ("install_helm_chart", {"repo_add"}, False),
    ("install_helm_chart", {"repo_update"}, False),
    ("install_helm_chart", {"with_repo"}, True),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("name,failures,success", HELM_WRITES, ids=[
    name + "-" + ("-".join(sorted(failures)) or "success")
    for name, failures, _ in HELM_WRITES
])
async def test_legacy_helm_commands_use_authoritative_environment(
    server, tmp_path, monkeypatch, name, failures, success
):
    from kubectl_mcp_tool.providers import KubernetesProvider
    from kubectl_mcp_tool.safety import SafetyMode, get_safety_mode, set_safety_mode

    authoritative = str(tmp_path / "authoritative.yaml")
    monkeypatch.setenv("MCP_K8S_KUBECONFIG", authoritative)
    monkeypatch.setenv("KUBECONFIG", str(tmp_path / "ambient.yaml"))
    monkeypatch.setattr(KubernetesProvider, "_instance", None)
    observed = []

    def check_output(command, **kwargs):
        if command[0] == "kubectl":
            phase = command[command.index("--context") + 2] + "_namespace"
        elif command[1:2] == ["repo"]:
            phase = "repo_" + command[2]
        else:
            phase = command[command.index("--kube-context") + 2]
        observed.append((phase, kwargs.get("env", {}).get("KUBECONFIG")))
        if phase in failures:
            raise subprocess.CalledProcessError(1, command, stderr="synthetic command failure")
        return "synthetic command succeeded"

    arguments = {"name": "web", "namespace": "synthetic", "context": "qa"}
    if name != "uninstall_helm_chart":
        arguments["chart"] = "web-chart"
    if failures & {"repo_add", "repo_update", "with_repo"}:
        arguments["repo"] = "synthetic=https://charts.invalid"
    previous_mode = get_safety_mode()
    set_safety_mode(SafetyMode.NORMAL)
    try:
        with patch("kubectl_mcp_tool.tools.helm.subprocess.check_output", side_effect=check_output):
            result = await call_protocol(server, name, arguments)
    finally:
        set_safety_mode(previous_mode)
    assert result["success"] is success
    if not success:
        assert "synthetic command failure" in result["error"]
    assert observed
    assert all(source == authoritative for _, source in observed), observed
    phases = [phase for phase, _ in observed]
    if "get_namespace" in failures:
        assert "create_namespace" in phases
    if "with_repo" in failures:
        assert phases[:2] == ["repo_add", "repo_update"]
