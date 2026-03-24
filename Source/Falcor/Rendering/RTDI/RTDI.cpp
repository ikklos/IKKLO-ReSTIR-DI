/***************************************************************************
 # Copyright (c) 2015-24, NVIDIA CORPORATION. All rights reserved.
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
#include "RTDI.h"
#include <algorithm>
#include <random>

namespace Falcor
{
    namespace
    {
        const std::string kShaderFile = "RenderPasses/ReSTIR_DI/ReSTIR.cs.slang";

        const char kInitEntry[] = "ReSTIR_Init";
        const char kVisibilityEntry[] = "ReSTIR_Visibility";
        const char kTemporalEntry[] = "ReSTIR_TemporalReuse";
        const char kSpatialEntry[] = "ReSTIR_SpatialReuse";
        const char kShadeEntry[] = "ReSTIR_ShadeColor";

        uint32_t getSeed()
        {
            static thread_local std::mt19937 mt{std::random_device{}()};
            static thread_local std::uniform_int_distribution<uint32_t> dist(0, UINT32_MAX);
            return dist(mt);
        }
    } // namespace

    RTDI::RTDI(ref<Scene> pScene, const Options& options)
        : mpScene(std::move(pScene))
        , mpDevice(mpScene ? mpScene->getDevice() : nullptr)
    {
        setOptions(options);
    }

    void RTDI::setOptions(const Options& options)
    {
        mOptions.risSampleCount = std::max(options.risSampleCount, 1u);
        mOptions.spatialReuseCount = std::max(options.spatialReuseCount, 1u);
        mOptions.debugShowWExplosion = options.debugShowWExplosion;
        mOptions.debugWThreshold = std::max(options.debugWThreshold, 0.1f);
        mResetHistory = true;
    }

    void RTDI::recreatePrograms()
    {
        mpInitPass = nullptr;
        mpVisibilityPass = nullptr;
        mpTemporalPass = nullptr;
        mpSpatialPass = nullptr;
        mpShadePass = nullptr;
    }

    ref<ComputePass> RTDI::createComputePass(const char* entryPoint) const
    {
        ProgramDesc desc;
        desc.addShaderModules(mpScene->getShaderModules());
        desc.addShaderLibrary(kShaderFile).csEntry(entryPoint);
        desc.addTypeConformances(mpScene->getTypeConformances());

        auto defines = mpScene->getSceneDefines();
        return ComputePass::create(mpDevice, desc, defines, true);
    }

    void RTDI::prepareBuffers(const ShaderVar& rootVar, uint32_t lightCount)
    {
        const uint32_t pixelCount = mFrameDim.x * mFrameDim.y;
        if (pixelCount != mReservoirElementCount || !mpReservoirBuffer || !mpPrevReservoirBuffer)
        {
            mReservoirElementCount = pixelCount;
            mpReservoirBuffer = mpDevice->createStructuredBuffer(rootVar["gRTDI"]["gReservoir"], mReservoirElementCount);
            mpPrevReservoirBuffer = mpDevice->createStructuredBuffer(rootVar["gRTDI"]["gPrevReservoir"], mReservoirElementCount);
            mpReservoirBuffer->setName("RTDI::Reservoir");
            mpPrevReservoirBuffer->setName("RTDI::PrevReservoir");
            mResetHistory = true;
        }

        const uint32_t aliasCount = std::max(lightCount, 1u);
        if (aliasCount != mAliasElementCount || !mpAliasProbBuffer || !mpAliasIndexBuffer || !mpLightPmfBuffer)
        {
            mAliasElementCount = aliasCount;
            mpAliasProbBuffer = mpDevice->createStructuredBuffer(rootVar["gRTDI"]["gAliasProb"], aliasCount);
            mpAliasIndexBuffer = mpDevice->createStructuredBuffer(rootVar["gRTDI"]["gAliasIndex"], aliasCount);
            mpLightPmfBuffer = mpDevice->createStructuredBuffer(rootVar["gRTDI"]["gLightPmf"], aliasCount);
            mpAliasProbBuffer->setName("RTDI::AliasProb");
            mpAliasIndexBuffer->setName("RTDI::AliasIndex");
            mpLightPmfBuffer->setName("RTDI::LightPmf");
        }
    }

    void RTDI::updateAliasTable(RenderContext* pRenderContext, const ref<LightCollection>& pLights, uint32_t lightCount)
    {
        const uint32_t aliasCount = std::max(lightCount, 1u);
        std::vector<float> probs(aliasCount, 1.f);
        std::vector<float> pmf(aliasCount, 1.f);
        std::vector<uint32_t> indices(aliasCount, 0u);

        if (lightCount > 0 && pLights)
        {
            pLights->prepareSyncCPUData(pRenderContext);
            const auto& triangles = pLights->getMeshLightTriangles(pRenderContext);
            const uint32_t triCount = std::min(lightCount, (uint32_t)triangles.size());

            std::vector<float> weights(lightCount, 0.f);
            float sumWeight = 0.f;
            for (uint32_t i = 0; i < triCount; ++i)
            {
                const float w = std::max(0.f, triangles[i].flux);
                weights[i] = w;
                sumWeight += w;
                indices[i] = i;
            }

            if (sumWeight > 0.f)
            {
                for (uint32_t i = 0; i < lightCount; ++i)
                    pmf[i] = weights[i] / sumWeight;

                std::vector<float> scaled(lightCount);
                std::vector<uint32_t> small;
                std::vector<uint32_t> large;
                small.reserve(lightCount);
                large.reserve(lightCount);

                const float scale = (float)lightCount / sumWeight;
                for (uint32_t i = 0; i < lightCount; ++i)
                {
                    scaled[i] = weights[i] * scale;
                    if (scaled[i] < 1.f)
                        small.push_back(i);
                    else
                        large.push_back(i);
                }

                while (!small.empty() && !large.empty())
                {
                    const uint32_t s = small.back();
                    small.pop_back();
                    const uint32_t l = large.back();

                    probs[s] = std::clamp(scaled[s], 0.f, 1.f);
                    indices[s] = l;

                    scaled[l] = (scaled[l] + scaled[s]) - 1.f;
                    if (scaled[l] < 1.f)
                    {
                        large.pop_back();
                        small.push_back(l);
                    }
                }

                for (uint32_t i : large)
                {
                    probs[i] = 1.f;
                    indices[i] = i;
                }
                for (uint32_t i : small)
                {
                    probs[i] = 1.f;
                    indices[i] = i;
                }
            }
            else
            {
                for (uint32_t i = 0; i < lightCount; ++i)
                {
                    probs[i] = 1.f;
                    pmf[i] = 1.f / std::max(lightCount, 1u);
                    indices[i] = i;
                }
            }
        }

        FALCOR_ASSERT(mpAliasProbBuffer && mpAliasIndexBuffer && mpLightPmfBuffer);
        mpAliasProbBuffer->setBlob(probs.data(), 0, sizeof(float) * aliasCount);
        mpAliasIndexBuffer->setBlob(indices.data(), 0, sizeof(uint32_t) * aliasCount);
        mpLightPmfBuffer->setBlob(pmf.data(), 0, sizeof(float) * aliasCount);
    }

    void RTDI::execute(
        RenderContext* pRenderContext,
        const ref<Texture>& pVBuffer,
        const ref<Texture>& pMotionVectors,
        const ref<Texture>& pColor
    )
    {
        if (!mpScene || !pVBuffer || !pMotionVectors || !pColor)
            return;

        mFrameDim = uint2(pVBuffer->getWidth(), pVBuffer->getHeight());

        const auto& pLights = mpScene->getLightCollection(pRenderContext);
        if (pLights)
            pLights->update(pRenderContext);

        if (is_set(mpScene->getUpdates(), IScene::UpdateFlags::RecompileNeeded) ||
            is_set(mpScene->getUpdates(), IScene::UpdateFlags::GeometryChanged))
        {
            recreatePrograms();
        }

        if (!mpInitPass)
            mpInitPass = createComputePass(kInitEntry);
        if (!mpVisibilityPass)
            mpVisibilityPass = createComputePass(kVisibilityEntry);
        if (!mpTemporalPass)
            mpTemporalPass = createComputePass(kTemporalEntry);
        if (!mpSpatialPass)
            mpSpatialPass = createComputePass(kSpatialEntry);
        if (!mpShadePass)
            mpShadePass = createComputePass(kShadeEntry);

        uint32_t lightCount = 0u;
        if (pLights)
        {
            pLights->prepareSyncCPUData(pRenderContext);
            lightCount = (uint32_t)pLights->getMeshLightTriangles(pRenderContext).size();
        }

        prepareBuffers(mpInitPass->getRootVar(), lightCount);

        if (mResetHistory || mFrameIndex == 0)
        {
            pRenderContext->clearUAV(mpPrevReservoirBuffer->getUAV().get(), uint4(0));
            mResetHistory = false;
        }

        updateAliasTable(pRenderContext, pLights, lightCount);

        auto bindCommonVars = [&](const ref<ComputePass>& pass, const ref<Buffer>& pReservoir = nullptr, const ref<Buffer>& pPrevReservoir = nullptr)
        {
            auto rootVar = pass->getRootVar();
            auto var = rootVar["gRTDI"];
            mpScene->bindShaderData(rootVar["gScene"]);

            var["gReservoir"] = pReservoir ? pReservoir : mpReservoirBuffer;
            var["gPrevReservoir"] = pPrevReservoir ? pPrevReservoir : mpPrevReservoirBuffer;
            var["gAliasProb"] = mpAliasProbBuffer;
            var["gAliasIndex"] = mpAliasIndexBuffer;
            var["gLightPmf"] = mpLightPmfBuffer;
            var["vbuffer"] = pVBuffer;
            var["mvec"] = pMotionVectors;

            auto cb = var;
            cb["gRIS_M"] = mOptions.risSampleCount;
            cb["gFrameDim"] = mFrameDim;
            cb["gFrameIndex"] = mFrameIndex;
            cb["gRandomSeed"] = getSeed();
            cb["gDebugShowWExplosion"] = mOptions.debugShowWExplosion ? 1u : 0u;
            cb["gDebugWThreshold"] = mOptions.debugWThreshold;
        };

        bindCommonVars(mpInitPass);
        mpInitPass->execute(pRenderContext, mFrameDim.x, mFrameDim.y);

        bindCommonVars(mpVisibilityPass);
        mpVisibilityPass->execute(pRenderContext, mFrameDim.x, mFrameDim.y);

        bindCommonVars(mpTemporalPass);
        mpTemporalPass->execute(pRenderContext, mFrameDim.x, mFrameDim.y);

        std::swap(mpReservoirBuffer, mpPrevReservoirBuffer);
        for (uint32_t i = 0; i < mOptions.spatialReuseCount; i++)
        {
            bindCommonVars(mpSpatialPass);
            mpSpatialPass->execute(pRenderContext, mFrameDim.x, mFrameDim.y);
            std::swap(mpReservoirBuffer, mpPrevReservoirBuffer);
        }

        bindCommonVars(mpShadePass, mpPrevReservoirBuffer, mpReservoirBuffer);
        mpShadePass->getRootVar()["gRTDI"]["color"] = pColor;
        mpShadePass->execute(pRenderContext, mFrameDim.x, mFrameDim.y);

        mFrameIndex++;
    }
}
