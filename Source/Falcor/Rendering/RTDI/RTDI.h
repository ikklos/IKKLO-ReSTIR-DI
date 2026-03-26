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
 ***************************************************************************/
#pragma once

#include "Falcor.h"

namespace Falcor
{
    /**
     * ReSTIR DI module wrapper.
     * Holds internal algorithm resources/passes and executes the full ReSTIR DI pipeline.
     */
    class FALCOR_API RTDI
    {
    public:
        struct Options
        {
            uint32_t risSampleCount = 8;
            uint32_t spatialReuseCount = 1;
            uint32_t presampledTileCount = 128;
            uint32_t presampledTileSize = 256;
            uint useEmissiveTextures = 0u;
            bool debugShowWExplosion = false;
            float debugWThreshold = 4.f;
        };

        RTDI(ref<Scene> pScene, const Options& options = Options());

        const Options& getOptions() const { return mOptions; }
        void setOptions(const Options& options);

        void execute(
            RenderContext* pRenderContext,
            const ref<Texture>& pVBuffer,
            const ref<Texture>& pMotionVectors,
            const ref<Texture>& pColor
        );

    private:
        struct LightRanges
        {
            uint32_t emissiveLightCount = 0;
            uint32_t localAnalyticLightCount = 0;
            uint32_t infiniteAnalyticLightCount = 0;
            bool envLightPresent = false;

            std::vector<uint32_t> analyticLightIDs;

            uint32_t getLocalLightCount() const { return emissiveLightCount + localAnalyticLightCount; }
            uint32_t getInfiniteLightCount() const { return infiniteAnalyticLightCount; }
            uint32_t getTotalLightCount() const { return getLocalLightCount() + getInfiniteLightCount() + (envLightPresent ? 1u : 0u); }
            uint32_t getFirstLocalLightIndex() const { return 0; }
            uint32_t getFirstLocalAnalyticLight() const { return emissiveLightCount; }
            uint32_t getFirstInfiniteLightIndex() const { return getLocalLightCount(); }
            uint32_t getEnvLightIndex() const { return getLocalLightCount() + getInfiniteLightCount(); }
        };

        void recreatePrograms();
        void prepareBuffers(const ShaderVar& rootVar, uint32_t lightCount);
        void updateLights(RenderContext* pRenderContext);
        void updateEnvLight(RenderContext* pRenderContext);
        void updateAliasTable(RenderContext* pRenderContext, const ref<LightCollection>& pLights, uint32_t lightCount);
        ref<ComputePass> createComputePass(const char* entryPoint) const;
        ref<ComputePass> createComputePass(const std::string& file, const std::string& entryPoint) const;

        ref<Scene> mpScene;
        ref<Device> mpDevice;

        ref<ComputePass> mpPresamplePass;
        ref<ComputePass> mpUpdateLightsPass;
        ref<ComputePass> mpUpdateEnvLightPass;
        ref<ComputePass> mpInitPass;
        ref<ComputePass> mpVisibilityPass;
        ref<ComputePass> mpTemporalPass;
        ref<ComputePass> mpSpatialPass;
        ref<ComputePass> mpShadePass;

        ref<Buffer> mpReservoirBuffer;
        ref<Buffer> mpPrevReservoirBuffer;
        ref<Buffer> mpAliasProbBuffer;
        ref<Buffer> mpAliasIndexBuffer;
        ref<Buffer> mpLightPmfBuffer;
        ref<Buffer> mpLightInfoBuffer;
        ref<Buffer> mpAnalyticLightIDBuffer;
        ref<Buffer> mpSurfaceDataBuffer;
        ref<Buffer> mpPresampledLightIndexBuffer;
        ref<Texture> mpLocalLightPdfTex;
        ref<Texture> mpEnvLightLuminanceTex;
        ref<Texture> mpEnvLightPdfTex;

        uint2 mFrameDim = uint2(0, 0);
        uint32_t mFrameIndex = 0;
        uint32_t mReservoirElementCount = 0;
        uint32_t mAliasElementCount = 0;
        uint32_t mPresampledLightElementCount = 0;
        bool mResetHistory = true;

        LightRanges mLights;
        Options mOptions;
    };
}
