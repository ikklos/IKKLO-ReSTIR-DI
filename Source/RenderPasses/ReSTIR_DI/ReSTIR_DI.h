/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/
#pragma once
#include "Falcor.h"
#include "RenderGraph/RenderPass.h"

using namespace Falcor;

class ReSTIR_DI : public RenderPass
{
public:
    FALCOR_PLUGIN_CLASS(ReSTIR_DI, "ReSTIR_DI", "Insert pass description here.");

    static ref<ReSTIR_DI> create(ref<Device> pDevice, const Properties& props)
    {
        return make_ref<ReSTIR_DI>(pDevice, props);
    }

    ReSTIR_DI(ref<Device> pDevice, const Properties& props);

    virtual Properties getProperties() const override;
    virtual RenderPassReflection reflect(const CompileData& compileData) override;
    virtual void compile(RenderContext* pRenderContext, const CompileData& compileData) override;
    virtual void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
    virtual void renderUI(Gui::Widgets& widget) override;
    virtual void setScene(RenderContext* pRenderContext, const ref<Scene>& pScene) override;
    virtual bool onMouseEvent(const MouseEvent& mouseEvent) override { return false; }
    virtual bool onKeyEvent(const KeyboardEvent& keyEvent) override { return false; }

private:
    void recreatePrograms();
    void prepareBuffers(const ShaderVar& rootVar, uint32_t lightCount);
    void updateAliasTable(RenderContext* pRenderContext, const ref<LightCollection>& pLights, uint32_t lightCount);

    ref<Scene> mpScene;
    ref<ComputePass> mpInitPass;
    ref<ComputePass> mpVisibilityPass;
    ref<ComputePass> mpTemporalPass;

    ref<Buffer> mpReservoirBuffer;
    ref<Buffer> mpPrevReservoirBuffer;
    ref<Buffer> mpSurfaceBuffer;
    ref<Buffer> mpAliasProbBuffer;
    ref<Buffer> mpAliasIndexBuffer;

    uint2 mFrameDim = uint2(0, 0);
    uint32_t mFrameIndex = 0;
    uint32_t mRISSampleCount = 8;
    uint32_t mReservoirElementCount = 0;
    uint32_t mAliasElementCount = 0;
};
