import asyncio
import base64
import json
import subprocess
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
import yaml

from kubectl_mcp_tool.tools.helm import register_helm_tools
from kubectl_mcp_tool.tools.operations import register_operations_tools


class ToolRegistry:
    def __init__(self):
        self.functions = {}

    def tool(self, **kwargs):
        def register(function):
            self.functions[function.__name__] = function
            return function
        return register


def install_tool():
    server = ToolRegistry()
    register_helm_tools(server, False, lambda: True)
    return server.functions['install_helm_chart']


def test_install_existing_namespace_never_queries_or_creates_namespace():
    commands = []
    def run(command, **kwargs):
        commands.append(command)
        if command[0] == 'kubectl':
            raise subprocess.CalledProcessError(1, command, stderr='Forbidden: cannot get namespaces')
        assert command == ['helm', '--kube-context', 'emr-engine-sit-sre', 'install', 'daft-history-server-wbw', './chart', '-n', 'weibowen']
        return 'STATUS: deployed'
    with patch('kubectl_mcp_tool.elicit.check_write_allowed', AsyncMock(return_value=None)), patch('kubectl_mcp_tool.tools.helm.get_kubectl_env', return_value={}), patch('kubectl_mcp_tool.tools.helm.subprocess.check_output', side_effect=run):
        result = asyncio.run(install_tool()(name='daft-history-server-wbw', chart='./chart', namespace='weibowen', context='emr-engine-sit-sre'))
    assert result['success'] is True
    assert len(commands) == 1


def test_namespace_creation_requires_explicit_opt_in():
    with patch('kubectl_mcp_tool.elicit.check_write_allowed', AsyncMock(return_value=None)), patch('kubectl_mcp_tool.tools.helm.get_kubectl_env', return_value={}), patch('kubectl_mcp_tool.tools.helm.subprocess.check_output', return_value='deployed') as run:
        result = asyncio.run(install_tool()(name='new', chart='./chart', namespace='new', context='test', create_namespace=True))
    assert result['success']
    assert run.call_count == 1
    assert '--create-namespace' in run.call_args.args[0]


def test_blocked_install_does_not_start_any_process():
    with patch('kubectl_mcp_tool.elicit.check_write_allowed', AsyncMock(return_value={'success': False, 'error': 'read-only'})), patch('kubectl_mcp_tool.tools.helm.subprocess.check_output') as run:
        result = asyncio.run(install_tool()(name='new', chart='./chart', namespace='existing', context='test'))
    assert result['success'] is False
    run.assert_not_called()


SECRET_VALUE = 'synthetic-test-signing-value'
SECRET_BASE64 = base64.b64encode(SECRET_VALUE.encode()).decode()


def secret():
    return {'apiVersion': 'v1', 'kind': 'Secret', 'metadata': {'name': 'signing', 'namespace': 'weibowen', 'uid': 'uid-1', 'resourceVersion': '7', 'annotations': {'kubectl.kubernetes.io/last-applied-configuration': json.dumps({'stringData': {'jwt': SECRET_VALUE}})}}, 'type': 'Opaque', 'data': {'jwt': SECRET_BASE64}, 'stringData': {'jwt': SECRET_VALUE}}


@pytest.mark.parametrize('format', ['json', 'yaml', 'yaml-stream'])
def test_generic_redacts_secret_values_and_annotations_in_mixed_readback(format):
    document = {'apiVersion': 'v1', 'kind': 'List', 'items': [secret(), {'apiVersion': 'v1', 'kind': 'Service', 'metadata': {'name': 'proxy'}}]}
    if format == 'json':
        raw = json.dumps(document)
    elif format == 'yaml':
        raw = yaml.safe_dump(document)
    else:
        raw = yaml.safe_dump_all(document['items'])
    server = ToolRegistry()
    register_operations_tools(server, False)
    with patch('kubectl_mcp_tool.tools.operations.get_kubectl_env', return_value={}), patch('kubectl_mcp_tool.tools.operations.subprocess.run', return_value=SimpleNamespace(returncode=0, stdout=raw, stderr='')):
        result = server.functions['kubectl_generic']('get service,secret -n weibowen -o '+('json' if format == 'json' else 'yaml'), context='emr-engine-sit-sre')
    serialized = json.dumps(result)
    assert SECRET_VALUE not in serialized
    assert SECRET_BASE64 not in serialized
    assert 'last-applied-configuration' not in serialized
    assert 'uid-1' in serialized and 'jwt' in serialized and 'proxy' in serialized
    assert result['success']


def test_generic_preserves_nonsecret_output_exactly():
    raw = '{"kind":"Service","metadata":{"name":"proxy"}}\n'
    server = ToolRegistry()
    register_operations_tools(server, False)
    with patch('kubectl_mcp_tool.tools.operations.get_kubectl_env', return_value={}), patch('kubectl_mcp_tool.tools.operations.subprocess.run', return_value=SimpleNamespace(returncode=0, stdout=raw, stderr='')):
        result = server.functions['kubectl_generic']('get service proxy -n weibowen -o json', context='emr-engine-sit-sre')
    assert result['output'] == raw


def test_secret_list_items_without_kind_are_redacted():
    from kubectl_mcp_tool.secret_readback import redact_secret_documents
    item = secret()
    item.pop('kind')
    item.pop('apiVersion')
    raw = json.dumps({'apiVersion': 'v1', 'kind': 'SecretList', 'items': [item]})
    output = redact_secret_documents(raw)
    assert SECRET_VALUE not in output and SECRET_BASE64 not in output
    assert 'last-applied-configuration' not in output
    assert 'uid-1' in output and 'jwt' in output


def test_timestamp_like_nonsecret_text_is_preserved():
    from kubectl_mcp_tool.secret_readback import redact_secret_documents
    assert redact_secret_documents('2026-99-99\n') == '2026-99-99\n'


def test_kind_label_in_normal_resource_does_not_trigger_redaction():
    from kubectl_mcp_tool.secret_readback import redact_secret_documents
    raw = '{"apiVersion":"v1","kind":"Service","metadata":{"name":"proxy","labels":{"kind":"Secret","stage":"qa"}}}\n'
    assert redact_secret_documents(raw) == raw


@pytest.mark.parametrize('mixed', [False, True])
def test_timestamp_like_scalar_cannot_disable_yaml_secret_redaction(mixed):
    from kubectl_mcp_tool.secret_readback import redact_secret_documents
    if mixed:
        raw = yaml.safe_dump(secret()) + '---\nkind: ConfigMap\nmetadata:\n  name: public\ndata:\n  date: 2026-99-99\n'
    else:
        raw = 'apiVersion: v1\nkind: Secret\nmetadata:\n  name: synthetic\nstringData:\n  password: 2026-99-99\n'
    output = redact_secret_documents(raw)
    assert SECRET_VALUE not in output and SECRET_BASE64 not in output
    if not mixed:
        assert '2026-99-99' not in output
    assert '[REDACTED]' in output


def test_secret_loader_does_not_change_global_yaml_date_behavior():
    from kubectl_mcp_tool.secret_readback import redact_secret_documents
    redact_secret_documents('2026-99-99\n')
    with pytest.raises(ValueError):
        yaml.safe_load('2026-99-99\n')


def test_secret_documents_are_redacted_from_error_field():
    server = ToolRegistry()
    register_operations_tools(server, False)
    raw = json.dumps({'kind': 'SecretList', 'items': [secret()]})
    with patch('kubectl_mcp_tool.tools.operations.get_kubectl_env', return_value={}), patch('kubectl_mcp_tool.tools.operations.subprocess.run', return_value=SimpleNamespace(returncode=1, stdout=raw, stderr=raw)):
        result = server.functions['kubectl_generic']('get secrets -n weibowen -o json', context='emr-engine-sit-sre')
    assert result['success'] is False
    assert result['error']
    assert SECRET_VALUE not in json.dumps(result) and SECRET_BASE64 not in json.dumps(result)


def test_secret_redaction_happens_before_mcp_serializes_both_content_channels():
    from fastmcp import Client, FastMCP
    server = FastMCP('secret-redaction-regression')
    register_operations_tools(server, False)
    async def exercise():
        with patch('kubectl_mcp_tool.tools.operations.get_kubectl_env', return_value={}), patch('kubectl_mcp_tool.tools.operations.subprocess.run', return_value=SimpleNamespace(returncode=0, stdout=json.dumps(secret()), stderr='')):
            async with Client(server) as session:
                result = await session.session.call_tool('kubectl_generic', {'command': 'get secret signing -n weibowen -o json', 'context': 'emr-engine-sit-sre'})
        serialized = result.model_dump_json()
        assert SECRET_VALUE not in serialized and SECRET_BASE64 not in serialized
        assert 'uid-1' in serialized
        assert not result.isError
    asyncio.run(exercise())
