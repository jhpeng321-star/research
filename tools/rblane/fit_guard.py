"""Restrict the registered own-project start loader BEFORE per-world reads."""
import ast


def guarded_start_source(code, worlds):
    if not worlds or len(set(worlds))!=len(worlds) or not set(worlds)<=set(range(969420,969432)):
        raise ValueError('unique development12 worlds required')
    tree=ast.parse(code)
    loops=[]
    for node in ast.walk(tree):
        if not isinstance(node,ast.For):continue
        target=any(isinstance(n,ast.Name) and n.id=='world' for n in ast.walk(node.target))
        iterator=node.iter
        if isinstance(iterator,ast.Call) and isinstance(iterator.func,ast.Name) and iterator.func.id=='enumerate':
            iterator=iterator.args[0]
        if target and isinstance(iterator,ast.Name) and iterator.id=='worlds':loops.append(node)
    if len(loops)!=1:raise ValueError('registered world-loop contract changed')
    guard=ast.parse(f"if world['seed'] not in {tuple(worlds)!r}:\n    continue").body[0]
    loops[0].body.insert(0,guard)
    ast.fix_missing_locations(tree)
    return tree,ast.unparse(tree)
