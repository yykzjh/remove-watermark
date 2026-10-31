import json
import struct

from watermark_studio.model_identity import compare_model_weights, weight_fingerprint


def store(root, parts):
    root.mkdir()
    mapping = {}
    for index, tensors in enumerate(parts):
        name = "model.safetensors" if len(parts) == 1 else f"model-{index}.safetensors"
        header, data = {}, b""
        for key, value in tensors.items():
            raw = struct.pack("<f", value)
            header[key] = {"dtype": "F32", "shape": [1], "data_offsets": [len(data), len(data)+len(raw)]}
            data += raw
            mapping[key] = name
        raw_header = json.dumps(header).encode()
        (root/name).write_bytes(struct.pack("<Q", len(raw_header))+raw_header+data)
    if len(parts)>1:
        (root/"model.safetensors.index.json").write_text(json.dumps({"weight_map": mapping}))
    return root


def test_identical_parameters_are_detected_across_different_sharding(tmp_path):
    editor = store(tmp_path/"editor", [{"a": 1, "b": 2}])
    verifier = store(tmp_path/"verifier", [{"b": 2}, {"a": 1}])
    result = compare_model_weights(editor, verifier)
    assert result["relation"] == "same_base_weights"
    assert result["editor_tensors"] == result["verifier_tensors"] == 2


def test_any_changed_tensor_invalidates_cached_identity(tmp_path):
    editor = store(tmp_path/"editor", [{"a": 1, "b": 2}])
    verifier = store(tmp_path/"verifier", [{"a": 1, "b": 2}])
    first = weight_fingerprint(verifier)
    file = verifier/"model.safetensors"
    contents = file.read_bytes()
    file.write_bytes(contents[:-4]+struct.pack("<f", 3))
    assert weight_fingerprint(verifier)["sha256"] != first["sha256"]
    assert compare_model_weights(editor, verifier)["relation"] == "different_weights"


def test_missing_or_truncated_checkpoints_never_assert_independence(tmp_path):
    editor = store(tmp_path/"editor", [{"a": 1}])
    assert compare_model_weights(editor, tmp_path/"missing")["relation"] == "unknown"
    (editor/"model.safetensors").write_bytes(b"broken")
    assert compare_model_weights(editor, editor)["relation"] == "unknown"


def test_index_cannot_escape_model_directory(tmp_path):
    root = tmp_path/"unsafe"
    root.mkdir()
    (root/"model.safetensors.index.json").write_text(json.dumps({"weight_map":{"a":"../private.safetensors"}}))
    assert compare_model_weights(root, root)["relation"] == "unknown"
