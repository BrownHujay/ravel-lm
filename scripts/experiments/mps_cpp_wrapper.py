"""Process-local workaround for missing MPS JIT wrapper shader getters."""
from torch._inductor.codegen.cpp_wrapper_mps import CppWrapperMps


def enable():
    original_define = CppWrapperMps.define_kernel

    def additional(self):
        emitted = getattr(self, "_ravel_emitted_getters", set())
        self._ravel_emitted_getters = emitted
        for name in dict.fromkeys(self.src_to_kernel.values()):
            if not name.startswith("mps_lib_") or name in emitted:
                continue
            emitted.add(name)
            self.prefix.splice(f'''
AOTIMetalKernelFunctionHandle get_{name}_handle() {{
    static auto kernel = []() {{
        AOTIMetalShaderLibraryHandle lib = nullptr;
        AOTIMetalKernelFunctionHandle fn = nullptr;
        AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_mps_create_shader_library({name}_source, &lib));
        AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_mps_get_kernel_function(lib, "generated_kernel", &fn));
        auto deleter = [](AOTIMetalShaderLibraryHandle h) {{
            if (h) aoti_torch_mps_delete_shader_library(h);
        }};
        using Owner = std::unique_ptr<AOTIMetalShaderLibraryOpaque, decltype(deleter)>;
        return std::make_pair(fn, Owner(lib, deleter));
    }}();
    return kernel.first;
}}
''')

    def define(self, *args, **kwargs):
        result = original_define(self, *args, **kwargs)
        additional(self)
        return result

    CppWrapperMps.define_kernel = define
    CppWrapperMps.codegen_additional_funcs = additional
