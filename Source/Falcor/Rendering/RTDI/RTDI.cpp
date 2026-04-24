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
 ***************************************************************************/
#include "RTDI.h"
#include <algorithm>
#include <random>

namespace Falcor
{
    namespace
    {
        const std::string kReSTIRShaderFile = "RenderPasses/ReSTIR_DI/ReSTIR.cs.slang";
        const std::string kLightUpdaterShaderFile = "Rendering/RTDI/LightUpdater.cs.slang";
        const std::string kEnvLightUpdaterShaderFile = "Rendering/RTDI/EnvLightUpdater.cs.slang";

        const char kPresampleEntry[] = "ReSTIR_Presample";
        const char kPresampleEnvEntry[] = "ReSTIR_PresampleEnv";
        const char kInitEntry[] = "ReSTIR_Init";
        const char kVisibilityEntry[] = "ReSTIR_Visibility";
        const char kTemporalEntry[] = "ReSTIR_TemporalReuse";
        const char kSpatialEntry[] = "ReSTIR_SpatialReuse";
        const char kSpatiotemporalEntry[] = "ReSTIR_SpatiotemporalReuse";
        const char kShadeEntry[] = "ReSTIR_ShadeColor";

        /*
         uint32_t getSeed()
        {
            static thread_local std::mt19937 mt{std::random_device{}()};
            static thread_local std::uniform_int_distribution<uint32_t> dist(0, UINT32_MAX);
            return dist(mt);
        }
        */
        uint32_t getSquareDim(uint32_t elementCount)
        {
            uint32_t dim = 1;
            while (dim * dim < std::max(elementCount, 1u)) dim <<= 1;
            return dim;
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
        mOptions.localRisSampleCount = std::min(options.localRisSampleCount, 64u);
        mOptions.infiniteRisSampleCount = std::min(options.infiniteRisSampleCount, 64u);
        mOptions.envRisSampleCount = std::min(options.envRisSampleCount, 64u);
        mOptions.brdfRisSampleCount = std::min(options.brdfRisSampleCount, 64u);
        mOptions.enableTemporalReuse = options.enableTemporalReuse;
        mOptions.enableSpatialReuse = options.enableSpatialReuse;
        mOptions.spatialReuseCount = std::max(options.spatialReuseCount, 1u);
        mOptions.spatialReuseSampleCount = std::clamp(options.spatialReuseSampleCount, 1u, 32u);
        mOptions.presampledTileCount = std::clamp(options.presampledTileCount, 1u, 1024u);
        mOptions.presampledTileSize = std::clamp(options.presampledTileSize, 1u, 8192u);
        mOptions.temporalSamplingRadius = std::clamp(options.temporalSamplingRadius, 0.f, 64.f);
        mOptions.spatialSamplingRadius = std::clamp(options.spatialSamplingRadius, 0.f, 64.f);
        mOptions.normalThreshold = std::clamp(options.normalThreshold, 0.f, 1.f);
        mOptions.depthThreshold = std::clamp(options.depthThreshold, 0.f, 1.f);
        mOptions.rayEpsilon = std::max(options.rayEpsilon, 1.0e-6f);
        mOptions.useEmissiveTextures = options.useEmissiveTextures;
        mResetHistory = true;
    }

    void RTDI::setGBufferAdjustShadingNormals(bool enabled)
    {
        if (mGBufferAdjustShadingNormals == enabled) return;
        mGBufferAdjustShadingNormals = enabled;
        recreatePrograms();
    }

    void RTDI::recreatePrograms()
    {
        mpPresamplePass = nullptr;
        mpPresampleEnvPass = nullptr;
        mpUpdateLightsPass = nullptr;
        mpUpdateEnvLightPass = nullptr;
        mpInitPass = nullptr;
        mpVisibilityPass = nullptr;
        mpTemporalPass = nullptr;
        mpSpatialPass = nullptr;
        mpSpatiotemporalPass = nullptr;
        mpShadePass = nullptr;
    }

    ref<ComputePass> RTDI::createComputePass(const char* entryPoint) const
    {
        return createComputePass(kReSTIRShaderFile, entryPoint);
    }

    ref<ComputePass> RTDI::createComputePass(const std::string& file, const std::string& entryPoint) const
    {
        ProgramDesc desc;
        desc.addShaderModules(mpScene->getShaderModules());
        desc.addShaderLibrary(file).csEntry(entryPoint);
        desc.addTypeConformances(mpScene->getTypeConformances());

        auto defines = mpScene->getSceneDefines();
        defines.add("GBUFFER_ADJUST_SHADING_NORMALS", mGBufferAdjustShadingNormals ? "1" : "0");
        return ComputePass::create(mpDevice, desc, defines, true);
    }

    void RTDI::prepareBuffers(const ShaderVar& rootVar)
    {
        const uint32_t pixelCount = mFrameDim.x * mFrameDim.y;
        if (pixelCount != mReservoirElementCount || !mpReservoirBuffer || !mpPrevReservoirBuffer || !mpSurfaceDataBuffer)
        {
            mReservoirElementCount = pixelCount;
            mpReservoirBuffer = mpDevice->createStructuredBuffer(rootVar["gRTDI"]["gReservoir"], mReservoirElementCount);
            mpPrevReservoirBuffer = mpDevice->createStructuredBuffer(rootVar["gRTDI"]["gPrevReservoir"], mReservoirElementCount);
            mpSurfaceDataBuffer = mpDevice->createStructuredBuffer(rootVar["gRTDI"]["surfaceData"], 2 * mReservoirElementCount);
            mpReservoirBuffer->setName("RTDI::Reservoir");
            mpPrevReservoirBuffer->setName("RTDI::PrevReservoir");
            mpSurfaceDataBuffer->setName("RTDI::SurfaceData");
            mResetHistory = true;
        }

        const uint32_t presampledLightCount = std::max(mOptions.presampledTileCount * mOptions.presampledTileSize, 1u);
        if (presampledLightCount != mPresampledLightElementCount || !mpPresampledLightIndexBuffer || !mpPresampledEnvDataBuffer)
        {
            mPresampledLightElementCount = presampledLightCount;
            mpPresampledLightIndexBuffer = mpDevice->createStructuredBuffer(rootVar["gRTDI"]["gPresampledLightIndex"], presampledLightCount);
            mpPresampledEnvDataBuffer = mpDevice->createStructuredBuffer(rootVar["gRTDI"]["gPresampledEnvData"], presampledLightCount);
            mpPresampledLightIndexBuffer->setName("RTDI::PresampledLightIndex");
            mpPresampledEnvDataBuffer->setName("RTDI::PresampledEnvData");
        }

        if(!mpNeighborOffsetBuffer){
            uint32_t elementCount = 2 * mNeighborOffsetCount;
            std::vector<uint8_t> offsets((size_t)elementCount);
            fillNeighborOffsetBuffer(offsets.data());
            mpNeighborOffsetBuffer = mpDevice->createTypedBuffer(
                ResourceFormat::RG8Snorm,
                mNeighborOffsetCount,
                ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
                MemoryType::DeviceLocal,
                offsets.data()
            );
        }
    }

    void RTDI::updateLights(RenderContext* pRenderContext)
    {
        FALCOR_ASSERT(mpInitPass && mpUpdateLightsPass);

        mLights.analyticLightIDs.clear();
        mLights.localAnalyticLightCount = 0;
        mLights.infiniteAnalyticLightCount = 0;

        if (mpScene->useAnalyticLights())
        {
            std::vector<uint32_t> localAnalytic;
            std::vector<uint32_t> infiniteAnalytic;
            for (uint32_t lightID = 0; lightID < mpScene->getActiveAnalyticLights().size(); ++lightID)
            {
                auto lightType = mpScene->getActiveAnalyticLights()[lightID]->getType();
                switch (lightType)
                {
                case LightType::Point: localAnalytic.push_back(lightID); break;
                case LightType::Directional:
                case LightType::Distant: infiniteAnalytic.push_back(lightID); break;
                default: break;
                }
            }

            mLights.localAnalyticLightCount = (uint32_t)localAnalytic.size();
            mLights.infiniteAnalyticLightCount = (uint32_t)infiniteAnalytic.size();
            mLights.analyticLightIDs.reserve(localAnalytic.size() + infiniteAnalytic.size());
            mLights.analyticLightIDs.insert(mLights.analyticLightIDs.end(), localAnalytic.begin(), localAnalytic.end());
            mLights.analyticLightIDs.insert(mLights.analyticLightIDs.end(), infiniteAnalytic.begin(), infiniteAnalytic.end());
        }

        auto pLightCollection = mpScene->getILightCollection(pRenderContext);
        mLights.emissiveLightCount = mpScene->useEmissiveLights() ? pLightCollection->getActiveLightCount(pRenderContext) : 0;
        mLights.envLightPresent = mpScene->useEnvLight();

        const uint32_t totalLightCount = mLights.getTotalLightCount();
        const uint32_t localLightCount = mLights.getLocalLightCount();

        auto rootVar = mpInitPass->getRootVar();

        if (totalLightCount > 0 && (!mpLightInfoBuffer || mpLightInfoBuffer->getElementCount() < totalLightCount))
        {
            mpLightInfoBuffer = mpDevice->createStructuredBuffer(rootVar["gRTDI"]["gLightInfo"], totalLightCount);
            mpLightInfoBuffer->setName("RTDI::LightInfo");
        }

        if (!mLights.analyticLightIDs.empty() && (!mpAnalyticLightIDBuffer || mpAnalyticLightIDBuffer->getElementCount() < mLights.analyticLightIDs.size()))
        {
            mpAnalyticLightIDBuffer = mpDevice->createStructuredBuffer(sizeof(uint32_t), (uint32_t)mLights.analyticLightIDs.size());
            mpAnalyticLightIDBuffer->setName("RTDI::AnalyticLightIDs");
        }

        if (mpAnalyticLightIDBuffer && !mLights.analyticLightIDs.empty())
            mpAnalyticLightIDBuffer->setBlob(mLights.analyticLightIDs.data(), 0, mLights.analyticLightIDs.size() * sizeof(uint32_t));

                {
            uint32_t dim = getSquareDim(std::max(localLightCount, 1u));
            if (!mpLocalLightPdfTex || mpLocalLightPdfTex->getWidth() != dim || mpLocalLightPdfTex->getHeight() != dim)
            {
                mpLocalLightPdfTex = mpDevice->createTexture2D(
                    dim,
                    dim,
                    ResourceFormat::R32Float,
                    1,
                    Resource::kMaxPossible,
                    nullptr,
                    ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::RenderTarget);
                mpLocalLightPdfTex->setName("RTDI::LocalLightPdf");
            }
        }

        if (totalLightCount > 0 && mpLightInfoBuffer)
        {
            if (mpLocalLightPdfTex) pRenderContext->clearUAV(mpLocalLightPdfTex->getUAV().get(), float4(0.f));

            uint2 threadCount(getSquareDim(totalLightCount), getSquareDim(totalLightCount));
            auto var = mpUpdateLightsPass->getRootVar()["gLightUpdater"];
            var["lightInfo"] = mpLightInfoBuffer;
            var["localLightPdf"] = mpLocalLightPdfTex;
            var["analyticLightIDs"] = mpAnalyticLightIDBuffer;

            var["threadCount"] = threadCount;
            var["totalLightCount"] = totalLightCount;
            var["firstLocalAnalyticLight"] = mLights.getFirstLocalAnalyticLight();
            var["firstInfiniteAnalyticLight"] = mLights.getFirstInfiniteLightIndex();
            var["envLightIndex"] = mLights.getEnvLightIndex();

            var["updateEmissiveLights"] = true;
            var["updateEmissiveLightsFlux"] = true;
            var["updateAnalyticLights"] = true;
            var["updateAnalyticLightsFlux"] = true;

            mpScene->bindShaderData(mpUpdateLightsPass->getRootVar()["gScene"]);
            mpUpdateLightsPass->execute(pRenderContext, threadCount.x, threadCount.y);

            if (mpLocalLightPdfTex) mpLocalLightPdfTex->generateMips(pRenderContext);
        }
    }

    void RTDI::updateEnvLight(RenderContext* pRenderContext)
    {
        FALCOR_ASSERT(mpUpdateEnvLightPass);

        if (!mpScene->useEnvLight())
        {
            if (mpEnvLightLuminanceTex) pRenderContext->clearUAV(mpEnvLightLuminanceTex->getUAV().get(), float4(0.f));
            if (mpEnvLightPdfTex) pRenderContext->clearUAV(mpEnvLightPdfTex->getUAV().get(), float4(0.f));
            return;
        }

        const auto& pEnvMap = mpScene->getEnvMap()->getEnvMap();
        if (!pEnvMap)
        {
            if (mpEnvLightLuminanceTex) pRenderContext->clearUAV(mpEnvLightLuminanceTex->getUAV().get(), float4(0.f));
            if (mpEnvLightPdfTex) pRenderContext->clearUAV(mpEnvLightPdfTex->getUAV().get(), float4(0.f));
            return;
        }

        uint32_t width = pEnvMap->getWidth();
        uint32_t height = pEnvMap->getHeight();

        if (!mpEnvLightLuminanceTex || mpEnvLightLuminanceTex->getWidth() != width || mpEnvLightLuminanceTex->getHeight() != height)
        {
            mpEnvLightLuminanceTex = mpDevice->createTexture2D(
                width, height, ResourceFormat::R32Float, 1, 1, nullptr,
                ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::RenderTarget);
            mpEnvLightLuminanceTex->setName("RTDI::EnvLuminance");
        }

        if (!mpEnvLightPdfTex || mpEnvLightPdfTex->getWidth() != width || mpEnvLightPdfTex->getHeight() != height)
        {
            mpEnvLightPdfTex = mpDevice->createTexture2D(
                width, height, ResourceFormat::R32Float, 1, Resource::kMaxPossible, nullptr,
                ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::RenderTarget);
            mpEnvLightPdfTex->setName("RTDI::EnvPdf");
        }

        auto var = mpUpdateEnvLightPass->getRootVar()["gEnvLightUpdater"];
        var["envLightLuminance"] = mpEnvLightLuminanceTex;
        var["envLightPdf"] = mpEnvLightPdfTex;
        var["texDim"] = uint2(width, height);
        mpScene->bindShaderData(mpUpdateEnvLightPass->getRootVar()["gScene"]);
        mpUpdateEnvLightPass->execute(pRenderContext, width, height);
        mpEnvLightPdfTex->generateMips(pRenderContext);
    }
    void RTDI::fillNeighborOffsetBuffer(uint8_t* buffer) const{
        int R = 250;
        const float phi2 = 1.0f / 1.3247179572447f;
        uint32_t num = 0;
        float u = 0.5f;
        float v = 0.5f;
        while (num < mNeighborOffsetCount * 2) {
            u += phi2;
            v += phi2 * phi2;
            if (u >= 1.0f) u -= 1.0f;
            if (v >= 1.0f) v -= 1.0f;

            float rSq = (u - 0.5f) * (u - 0.5f) + (v - 0.5f) * (v - 0.5f);
            if (rSq > 0.25f)
                continue;

            buffer[num++] = int8_t((u - 0.5f) * R);
            buffer[num++] = int8_t((v - 0.5f) * R);
        }
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
        if (mFrameIndex == 0) mPrevCameraData = mpScene->getCamera()->getData();

        const auto& pLights = mpScene->getLightCollection(pRenderContext);
        if (pLights) pLights->update(pRenderContext);

        if (is_set(mpScene->getUpdates(), IScene::UpdateFlags::RecompileNeeded) ||
            is_set(mpScene->getUpdates(), IScene::UpdateFlags::GeometryChanged))
        {
            recreatePrograms();
        }

        if (!mpUpdateLightsPass) mpUpdateLightsPass = createComputePass(kLightUpdaterShaderFile, "main");
        if (!mpUpdateEnvLightPass) mpUpdateEnvLightPass = createComputePass(kEnvLightUpdaterShaderFile, "main");
        if (!mpPresamplePass) mpPresamplePass = createComputePass(kPresampleEntry);
        if (!mpPresampleEnvPass) mpPresampleEnvPass = createComputePass(kPresampleEnvEntry);
        if (!mpInitPass) mpInitPass = createComputePass(kInitEntry);
        if (!mpVisibilityPass) mpVisibilityPass = createComputePass(kVisibilityEntry);
        if (!mpTemporalPass) mpTemporalPass = createComputePass(kTemporalEntry);
        if (!mpSpatialPass) mpSpatialPass = createComputePass(kSpatialEntry);
        if (!mpSpatiotemporalPass) mpSpatiotemporalPass = createComputePass(kSpatiotemporalEntry);
        if (!mpShadePass) mpShadePass = createComputePass(kShadeEntry);

        updateLights(pRenderContext);
        updateEnvLight(pRenderContext);

        prepareBuffers(mpInitPass->getRootVar());

        const uint32_t frameSeed = (mFrameIndex * 1664525u + 1013904223u) ^ 0xA511E9B3u;

        auto bindCommonVars = [&](const ref<ComputePass>& pass, uint32_t passSalt, const ref<Buffer>& pReservoir = nullptr, const ref<Buffer>& pPrevReservoir = nullptr)
        {
            pass->addDefine("GBUFFER_ADJUST_SHADING_NORMALS", mGBufferAdjustShadingNormals ? "1" : "0");
            auto rootVar = pass->getRootVar();
            auto var = rootVar["gRTDI"];
            mpScene->bindShaderData(rootVar["gScene"]);

            var["gReservoir"] = pReservoir ? pReservoir : mpReservoirBuffer;
            var["gPrevReservoir"] = pPrevReservoir ? pPrevReservoir : mpPrevReservoirBuffer;
            var["gLightInfo"] = mpLightInfoBuffer;
            var["surfaceData"] = mpSurfaceDataBuffer;
            var["gPresampledLightIndex"] = mpPresampledLightIndexBuffer;
            var["gPresampledEnvData"] = mpPresampledEnvDataBuffer;
            var["vbuffer"] = pVBuffer;
            var["mvec"] = pMotionVectors;
            var["localLightPdfTexture"] = mpLocalLightPdfTex;
            var["envLightLuminanceTexture"] = mpEnvLightLuminanceTex;
            var["envLightPdfTexture"] = mpEnvLightPdfTex;
            var["gNeighborOffsetBuffer"] = mpNeighborOffsetBuffer;
            uint32_t currentSurfaceBufferIndex = mFrameIndex & 1u;
            auto cb = var;
            cb["gLocalRIS_M"] = mOptions.localRisSampleCount;
            cb["gInfiniteRIS_M"] = mOptions.infiniteRisSampleCount;
            cb["gEnvRIS_M"] = mOptions.envRisSampleCount;
            cb["gBrdfRIS_M"] = mOptions.brdfRisSampleCount;
            cb["gSpatialReuseSampleCount"] = mOptions.spatialReuseSampleCount;
            cb["gPresampledTileCount"] = mOptions.presampledTileCount;
            cb["gPresampledTileSize"] = mOptions.presampledTileSize;
            cb["gTemporalSamplingRadius"] = mOptions.temporalSamplingRadius;
            cb["gSpatialSamplingRadius"] = mOptions.spatialSamplingRadius;
            cb["gFrameDim"] = mFrameDim;
            cb["gFrameIndex"] = mFrameIndex;
            cb["gRandomSeed"] = frameSeed ^ passSalt;
            cb["normalThreshold"] = mOptions.normalThreshold;
            cb["depthThreshold"] = mOptions.depthThreshold;
            cb["rayEpsilon"] = mOptions.rayEpsilon;
            cb["useEmissiveTextures"] = mOptions.useEmissiveTextures;

            cb["firstLocalAnalyticLight"] = mLights.getFirstLocalAnalyticLight();
            cb["firstInfiniteAnalyticLight"] = mLights.getFirstInfiniteLightIndex();
            cb["envLightIndex"] = mLights.getEnvLightIndex();
            cb["totalLightCount"] = mLights.getTotalLightCount();

            cb["currentSurfaceBufferIndex"] = currentSurfaceBufferIndex;
            cb["prevSurfaceBufferIndex"] = 1u - currentSurfaceBufferIndex;
            cb["gNeighborOffsetMask"] = mNeighborOffsetCount - 1;
            cb["prevCameraU"] = mPrevCameraData.cameraU;
            cb["prevCameraV"] = mPrevCameraData.cameraV;
            cb["prevCameraW"] = mPrevCameraData.cameraW;
            cb["prevCameraJitter"] = float2(mPrevCameraData.jitterX, mPrevCameraData.jitterY);
        };

        if (mResetHistory || mFrameIndex == 0)
        {
            pRenderContext->clearUAV(mpPrevReservoirBuffer->getUAV().get(), uint4(0u));
            mResetHistory = false;
        }

        bindCommonVars(mpPresamplePass, 0x13579BDFu);
        const uint32_t presampleCount = std::max(mOptions.presampledTileCount * mOptions.presampledTileSize, 1u);
        mpPresamplePass->execute(pRenderContext, presampleCount, 1);

        bindCommonVars(mpPresampleEnvPass, 0x55AA11EEu);
        mpPresampleEnvPass->execute(pRenderContext, presampleCount, 1);

        bindCommonVars(mpInitPass, 0x2468ACE0u);
        mpInitPass->execute(pRenderContext, mFrameDim.x, mFrameDim.y);

        bindCommonVars(mpVisibilityPass, 0x31415926u);
        mpVisibilityPass->execute(pRenderContext, mFrameDim.x, mFrameDim.y);

        ref<Buffer> pFinalReservoir = mpReservoirBuffer;

        if (mOptions.enableTemporalReuse && mOptions.enableSpatialReuse)
        {
            bindCommonVars(mpSpatiotemporalPass, 0x6C8E9CF5u);
            mpSpatiotemporalPass->execute(pRenderContext, mFrameDim.x, mFrameDim.y);
            pFinalReservoir = mpReservoirBuffer;
        }
        else if (mOptions.enableTemporalReuse)
        {
            bindCommonVars(mpTemporalPass, 0x27182818u);
            mpTemporalPass->execute(pRenderContext, mFrameDim.x, mFrameDim.y);
            pFinalReservoir = mpReservoirBuffer;
        }

        if (mOptions.enableSpatialReuse && !mOptions.enableTemporalReuse)
        {
            // Prepare ping-pong for spatial reuse: input from mpPrevReservoirBuffer, output to mpReservoirBuffer.
            if (pFinalReservoir == mpReservoirBuffer) std::swap(mpReservoirBuffer, mpPrevReservoirBuffer);

            for (uint32_t i = 0; i < mOptions.spatialReuseCount; i++)
            {
                bindCommonVars(mpSpatialPass, 0x5A5A5A5Au ^ i);
                mpSpatialPass->execute(pRenderContext, mFrameDim.x, mFrameDim.y);
                std::swap(mpReservoirBuffer, mpPrevReservoirBuffer);
            }

            pFinalReservoir = mpPrevReservoirBuffer;
        }

        bindCommonVars(mpShadePass, 0xC001D00Du, pFinalReservoir, mpReservoirBuffer);
        mpShadePass->getRootVar()["gRTDI"]["color"] = pColor;
        mpShadePass->execute(pRenderContext, mFrameDim.x, mFrameDim.y);

        // Keep last frame final reservoirs in mpPrevReservoirBuffer for temporal reuse next frame.
        if (pFinalReservoir == mpReservoirBuffer) std::swap(mpReservoirBuffer, mpPrevReservoirBuffer);

        mPrevCameraData = mpScene->getCamera()->getData();
        mFrameIndex++;
    }
}
