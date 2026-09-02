import os
import sys

# 手動編譯安裝路徑
OV_ROOT = "/opt/openvino_2022.3.1"

def initialize_openvino():
    if not os.path.exists(OV_ROOT):
        print(f"找不到編譯後的 OpenVINO: {OV_ROOT}")
        return False
    
    ov_python_path = os.path.join(OV_ROOT, "python/python3.10")
    if os.path.exists(ov_python_path):
        if ov_python_path not in sys.path:
            sys.path.insert(0, ov_python_path)    

    runtime_lib = os.path.join(OV_ROOT, "runtime/lib/aarch64")
    if os.path.exists(runtime_lib):
        os.environ['LD_LIBRARY_PATH'] = runtime_lib + ":" + os.environ.get('LD_LIBRARY_PATH', '')
    
    return True

# 執行初始化
if initialize_openvino():
    try:
        import openvino.runtime as _ov_runtime
        
        # Monkey Patch
        import types
        fake_openvino = types.ModuleType("openvino")
        fake_openvino.Core = _ov_runtime.Core
        fake_openvino.runtime = _ov_runtime
        sys.modules["openvino"] = fake_openvino
    except ImportError as e:
        print(f"錯誤訊息: {e}")
