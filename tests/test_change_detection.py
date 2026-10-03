"""Change detection must key components by exact analyzer id, not name."""

from repodoc.agent.update import ChangeType, DiffSpec, analyze_diff, _component_id


def _changes(analysis):
    return {(c.component_id, c.change_type) for c in analysis.changes}


def test_java_same_named_methods_do_not_collapse():
    old = (
        "public class A {\n"
        "  public void save(int x) {}\n"
        "}\n"
        "public class B {\n"
        "  public void save(int x) {}\n"
        "}\n"
    )
    new = (
        "public class A {\n"
        "  public void save(int x, String y) {}\n"
        "}\n"
    )
    result = _changes(analyze_diff("m.java", old, new, DiffSpec("java")))
    assert ("m.B.save", ChangeType.REMOVED_COMPONENT) in result
    assert ("m.A.save", ChangeType.API_SIGNATURE_CHANGED) in result


def test_go_method_signature_change_is_severe():
    old = (
        "package p\n\n"
        "type User struct{}\n\n"
        "func (u *User) Save(name string) error { return nil }\n"
    )
    new = old.replace("Save(name string)", "Save(name string, age int)")
    analysis = analyze_diff("m.go", old, new, DiffSpec("go"))
    change = next(c for c in analysis.changes if c.name == "Save")
    assert change.change_type == ChangeType.API_SIGNATURE_CHANGED
    assert change.component_id  # exact graph node id, not a name-derived guess


def test_rust_signature_change_is_severe():
    old = "pub struct S;\n\nimpl S {\n    pub fn run(&self, x: i32) {}\n}\n"
    new = old.replace("x: i32", "x: i32, y: i32")
    analysis = analyze_diff("m.rs", old, new, DiffSpec("rust"))
    change = next(c for c in analysis.changes if c.name == "run")
    assert change.change_type == ChangeType.API_SIGNATURE_CHANGED


def test_component_id_fallback_strips_extension():
    assert _component_id("src/pkg/mod.py", "func") == "src.pkg.mod.func"
    assert _component_id("src/a.tsx", "App") == "src.a.App"
