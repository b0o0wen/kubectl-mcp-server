"""Metadata-only projections for structured Kubernetes Secret output."""

import json

import yaml


class _ResourceLoader(yaml.SafeLoader):
    # Kubernetes timestamp fields are strings. Keep date-like values as text,
    # without changing SafeLoader's global resolver/constructor dictionaries.
    yaml_implicit_resolvers = {
        key: [resolver for resolver in resolvers if resolver[0] != 'tag:yaml.org,2002:timestamp']
        for key, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
    }
    yaml_constructors = dict(yaml.SafeLoader.yaml_constructors)
    yaml_constructors['tag:yaml.org,2002:timestamp'] = yaml.SafeLoader.construct_scalar


def redact_secret_documents(text: str) -> str:
    """Redact Secret JSON/YAML documents, leaving other output unchanged.

    This handles full resource documents, including mixed Lists and YAML
    streams. Arbitrary jsonpath/template output is not a resource document;
    callers must request metadata projections for those output formats.
    """
    try:
        documents = [json.loads(text)]
        is_json = True
    except (ValueError, TypeError):
        try:
            documents = list(yaml.load_all(text, Loader=_ResourceLoader))
        except (yaml.YAMLError, ValueError):
            return text
        is_json = False

    seen = set()

    def redact(node, secret_item=False):
        if not isinstance(node, dict):
            return False
        if secret_item or node.get('kind') == 'Secret':
            metadata = node.get('metadata') or {}
            safe = {key: node[key] for key in ('apiVersion', 'kind', 'type') if key in node}
            safe['metadata'] = {key: metadata[key] for key in ('name', 'namespace', 'uid', 'resourceVersion', 'creationTimestamp') if key in metadata}
            for field in ('data', 'stringData'):
                if isinstance(node.get(field), dict):
                    safe[field] = {key: '[REDACTED]' for key in node[field]}
            node.clear()
            node.update(safe)
            return True
        if id(node) in seen:
            return False
        seen.add(id(node))
        changed = False
        if node.get('kind') in ('List', 'SecretList') and isinstance(node.get('items'), list):
            for item in node['items']:
                changed = redact(item, secret_item=node['kind'] == 'SecretList') or changed
        elif node.get('type') in ('ADDED', 'MODIFIED', 'DELETED', 'BOOKMARK', 'ERROR'):
            changed = redact(node.get('object'))
        return changed

    changed = False
    for document in documents:
        changed = redact(document) or changed
    if not changed:
        return text
    if is_json:
        return json.dumps(documents[0], ensure_ascii=False, indent=2)
    return yaml.safe_dump_all(documents, sort_keys=False)
