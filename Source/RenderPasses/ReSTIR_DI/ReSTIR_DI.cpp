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
#include "ReSTIR_DI.h"

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, ReSTIR_DI>();
}

namespace
{
const std::string kShaderFile = "RenderPasses/ReSTIR_DI/ReSTIR.cs.slang";
const char kInitEntry[] = "ReSTIR_Init";
const char kVisibilityEntry[] = "ReSTIR_Visibility";
const char kInputVBuffer[] = "vbuffer";
const char kRISSampleCount[] = "risSampleCount";
}

ReSTIR_DI::ReSTIR_DI(ref<Device> pDevice, const Properties& props) : RenderPass(pDevice)
{
    for (const auto& [key, value] : props)
    {
        if (key == kRISSampleCount)
            mRISSampleCount = value;
        else
            logWarning("Unknown property '{}' in ReSTIR_DI properties.", key);
    }
}

Properties ReSTIR_DI::getProperties() const
{
    Properties props;
    props[kRISSampleCount] = mRISSampleCount;
    return props;
}

RenderPassReflection ReSTIR_DI::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;
    reflector.addInput(kInputVBuffer, "Visibility buffer in packed format");
    return reflector;
}

void ReSTIR_DI::compile(RenderContext* pRenderContext, const CompileData& compileData)
{
    mFrameDim = compileData.defaultTexDims;
}

void ReSTIR_DI::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    mpScene = pScene;
    mFrameIndex = 0;
    recreatePrograms();

    mpReservoirBuffer = nullptr;
    mpPrevReservoirBuffer = nullptr;
    mpAliasProbBuffer = nullptr;
    mpAliasIndexBuffer = nullptr;
    mReservoirElementCount = 0;
    mAliasElementCount = 0;
}

void ReSTIR_DI::recreatePrograms()
{
    mpInitPass = nullptr;
    mpVisibilityPass = nullptr;
}

void ReSTIR_DI::prepareBuffers(const ShaderVar& rootVar, uint32_t lightCount)
{
    const uint32_t pixelCount = mFrameDim.x * mFrameDim.y;
    if (pixelCount != mReservoirElementCount || !mpReservoirBuffer || !mpPrevReservoirBuffer)
    {
        mReservoirElementCount = pixelCount;
        mpReservoirBuffer = mpDevice->createStructuredBuffer(rootVar["gReservoir"], mReservoirElementCount);
        mpPrevReservoirBuffer = mpDevice->createStructuredBuffer(rootVar["gPrevReservoir"], mReservoirElementCount);
        mpReservoirBuffer->setName("ReSTIR_DI::Reservoir");
        mpPrevReservoirBuffer->setName("ReSTIR_DI::PrevReservoir");
    }

    const uint32_t aliasCount = std::max(lightCount, 1u);
    if (aliasCount != mAliasElementCount || !mpAliasProbBuffer || !mpAliasIndexBuffer)
    {
        std::vector<float> probs(aliasCount, 1.f);
        std::vector<uint32_t> indices(aliasCount, 0u);
        for (uint32_t i = 0; i < aliasCount; ++i) indices[i] = i;

        mAliasElementCount = aliasCount;
        mpAliasProbBuffer = mpDevice->createTypedBuffer<float>(aliasCount, ResourceBindFlags::ShaderResource, MemoryType::DeviceLocal, probs.data());
        mpAliasIndexBuffer = mpDevice->createTypedBuffer<uint32_t>(aliasCount, ResourceBindFlags::ShaderResource, MemoryType::DeviceLocal, indices.data());
        mpAliasProbBuffer->setName("ReSTIR_DI::AliasProb");
        mpAliasIndexBuffer->setName("ReSTIR_DI::AliasIndex");
    }
}

void ReSTIR_DI::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    if (!mpScene) return;

    auto pVBuffer = renderData.getTexture(kInputVBuffer);
    if (!pVBuffer) return;

    mFrameDim = uint2(pVBuffer->getWidth(), pVBuffer->getHeight());

    // Keep emissive light data up to date for gScene.lightCollection usage in shader.
    const auto& pLights = mpScene->getLightCollection(pRenderContext);
    if (pLights) pLights->update(pRenderContext);

    if (is_set(mpScene->getUpdates(), IScene::UpdateFlags::RecompileNeeded) ||
        is_set(mpScene->getUpdates(), IScene::UpdateFlags::GeometryChanged))
    {
        recreatePrograms();
    }

    if (!mpInitPass)
    {
        ProgramDesc desc;
        desc.addShaderModules(mpScene->getShaderModules());
        desc.addShaderLibrary(kShaderFile).csEntry(kInitEntry);
        desc.addTypeConformances(mpScene->getTypeConformances());

        auto defines = mpScene->getSceneDefines();
        mpInitPass = ComputePass::create(mpDevice, desc, defines, true);
    }

    if (!mpVisibilityPass)
    {
        ProgramDesc desc;
        desc.addShaderModules(mpScene->getShaderModules());
        desc.addShaderLibrary(kShaderFile).csEntry(kVisibilityEntry);
        desc.addTypeConformances(mpScene->getTypeConformances());

        auto defines = mpScene->getSceneDefines();
        mpVisibilityPass = ComputePass::create(mpDevice, desc, defines, true);
    }

    auto bindCommonVars = [&](const ref<ComputePass>& pass)
    {
        auto rootVar = pass->getRootVar();
        mpScene->bindShaderData(rootVar["gScene"]);

        rootVar["gReservoir"] = mpReservoirBuffer;
        rootVar["gPrevReservoir"] = mpPrevReservoirBuffer;
        rootVar["gAliasProb"] = mpAliasProbBuffer;
        rootVar["gAliasIndex"] = mpAliasIndexBuffer;
        rootVar["vbuffer"] = pVBuffer;

        auto cb = rootVar["CB"];
        cb["gRIS_M"] = mRISSampleCount;
        cb["gFrameDim"] = float2((float)mFrameDim.x, (float)mFrameDim.y);
        cb["gFrameIndex"] = mFrameIndex;
    };

    const uint32_t lightCount = pLights ? pLights->getTotalLightCount() : 0u;
    prepareBuffers(mpInitPass->getRootVar(), lightCount);

    bindCommonVars(mpInitPass);
    mpInitPass->execute(pRenderContext, mFrameDim.x, mFrameDim.y);

    bindCommonVars(mpVisibilityPass);
    mpVisibilityPass->execute(pRenderContext, mFrameDim.x, mFrameDim.y);

    std::swap(mpReservoirBuffer, mpPrevReservoirBuffer);
    mFrameIndex++;
}

void ReSTIR_DI::renderUI(Gui::Widgets& widget)
{
    widget.var("RIS M", mRISSampleCount, 1u, 64u);
}
