module derelict.cudnn7;

/**
    Translation of cudnn.h
    forked from https://github.com/henrygouk/DerelictCuDNN/blob/master/source/derelict/cudnn7.d
*/

import derelict.cuda.runtimeapi;
import derelict.util.loader;

private
{
    import derelict.util.system;

    static if(Derelict_OS_Linux)
    {
        version(X86_64)
            enum libNames = "libcudnn.so.7";
        else
            static assert(0, "Need to implement cuDNN libNames for this arch.");
    }
    else static if(Derelict_OS_Windows)
    {
        version(X86_64)
            enum libNames = "cudnn64_7.dll";
        else
            static assert(0, "There are no cuDNN libNames for this arch and operating system.");
    }
    else static if(Derelict_OS_Mac)
    {
        version(X86_64)
            enum libNames = "libcudnn.7.dylib,libcudnn.dylib";
        else
            static assert(0, "There are no cuDNN libNames for this arch and operating system.");
    }
    else
    {
        static assert(0, "Need to implement cuDNN libNames for this operating system.");
    }

    enum functionTypes = [
        ["cudnnCreate", "cudnnHandle_t *"],
        ["cudnnDestroy", "cudnnHandle_t"],
        ["cudnnSetStream", "cudnnHandle_t", "cudaStream_t"],
        ["cudnnGetStream", "cudnnHandle_t", "cudaStream_t *"],


        ["cudnnCreateTensorDescriptor", "cudnnTensorDescriptor_t *"],
        ["cudnnSetTensor4dDescriptor", "cudnnTensorDescriptor_t", "cudnnTensorFormat_t", "cudnnDataType_t", "int",
            "int", "int", "int"],
        ["cudnnSetTensor4dDescriptorEx", "cudnnTensorDescriptor_t", "cudnnDataType_t", "int", "int", "int", "int",
            "int", "int", "int", "int"],
        ["cudnnGetTensor4dDescriptor", "const cudnnTensorDescriptor_t", "cudnnDataType_t *", "int *", "int *", "int *",
            "int *", "int *", "int *", "int *", "int *"],
        ["cudnnSetTensorNdDescriptor", "cudnnTensorDescriptor_t", "cudnnDataType_t", "int", "int *", "int *"],
        ["cudnnGetTensorNdDescriptor", "const cudnnTensorDescriptor_t", "int", "cudnnDataType_t *", "int *", "int",
            "int"],
        ["cudnnDestroyTensorDescriptor", "cudnnTensorDescriptor_t"],
        ["cudnnTransformTensor", "cudnnHandle_t", "const void *", "const cudnnTensorDescriptor_t", "const void *",
            "const void *", "const cudnnTensorDescriptor_t", "void *"],
        ["cudnnAddTensor", "cudnnHandle_t", "const void *", "const cudnnTensorDescriptor_t", "const void *",
            "const void *", "const cudnnTensorDescriptor_t", "void *"],
        

        ["cudnnCreateOpTensorDescriptor", "cudnnOpTensorDescriptor_t *"],
        ["cudnnSetOpTensorDescriptor", "cudnnOpTensorDescriptor_t", "cudnnOpTensorOp_t", "cudnnDataType_t",
            "cudnnNanPropagation_t"],
        ["cudnnGetOpTensorDescriptor", "const cudnnOpTensorDescriptor_t", "cudnnOpTensorOp_t *", "cudnnDataType_t *",
            "cudnnNanPropagation_t *"],
        ["cudnnDestroyOpTensorDescriptor", "cudnnOpTensorDescriptor_t"],
        ["cudnnOpTensor", "cudnnHandle_t", "const cudnnOpTensorDescriptor_t", "const void *",
            "const cudnnTensorDescriptor_t", "const void *", "const void *", "const cudnnTensorDescriptor_t",
            "const void *", "const void *", "const cudnnTensorDescriptor_t", "void *"],
        ["cudnnSetTensor", "cudnnHandle_t", "const cudnnTensorDescriptor_t", "void *", "const void *"],
        ["cudnnScaleTensor", "cudnnHandle_t", "const cudnnTensorDescriptor_t", "void *", "const void *"],
        

        ["cudnnCreateFilterDescriptor", "cudnnFilterDescriptor_t *"],
        ["cudnnSetFilter4dDescriptor", "cudnnFilterDescriptor_t", "cudnnDataType_t", "cudnnTensorFormat_t", "int",
			"int", "int", "int"],
        ["cudnnGetFilter4dDescriptor", "cudnnDataType_t *", "cudnnTensorFormat_t *", "int *", "int *", "int *",
			"int *"],
        ["cudnnSetFilterNdDescriptor", "cudnnFilterDescriptor_t", "cudnnDataType_t", "cudnnTensorFormat_t", "int",
            "const int[]"],
        ["cudnnGetFilterNdDescriptor", "const cudnnFilterDescriptor_t", "int", "cudnnDataType_t *",
            "cudnnTensorFormat_t *", "int *", "int[]"],
        ["cudnnDestroyFilterDescriptor", "cudnnFilterDescriptor_t"],
        ["cudnnCreateConvolutionDescriptor", "cudnnConvolutionDescriptor_t *"],
        ["cudnnSetConvolution2dDescriptor", "cudnnConvolutionDescriptor_t", "int", "int", "int", "int", "int",
            "int", "cudnnConvolutionMode_t", "cudnnDataType_t"],
        ["cudnnGetConvolution2dDescriptor", "const cudnnConvolutionDescriptor_t", "int *", "int *", "int *",
            "int *", "int *", "int *", "cudnnConvolutionMode_t *", "cudnnDataType_t *"],
        ["cudnnGetConvolution2dForwardOutputDim", "const cudnnConvolutionDescriptor_t",
            "const cudnnTensorDescriptor_t", "const cudnnFilterDescriptor_t", "int", "int", "int", "int"],
        ["cudnnSetConvolutionNdDescriptor", "cudnnConvolutionDescriptor_t", "int", "const int[]", "const int[]",
            "const int[]", "cudnnConvolutionMode_t", "cudnnDataType_t"],
        ["cudnnGetConvolutionNdDescriptor", "const cudnnConvolutionDescriptor_t", "int", "int *", "int[]", "int[]",
            "int[]", "cudnnConvolutionMode_t *", "cudnnDataType_t *"],
        ["cudnnGetConvolutionNdForwardOutputDim", "const cudnnConvolutionDescriptor_t",
            "const cudnnTensorDescriptor_t", "const cudnnFilterDescriptor_t", "int", "int[]"],
        ["cudnnDestroyConvolutionDescriptor", "cudnnConvolutionDescriptor_t"],

        
        ["cudnnFindConvolutionForwardAlgorithm", "cudnnHandle_t", "const cudnnTensorDescriptor_t",
            "const cudnnFilterDescriptor_t", "const cudnnConvolutionDescriptor_t", "const cudnnTensorDescriptor_t",
            "const int", "int *", "cudnnConvolutionFwdAlgoPerf_t *"],
        ["cudnnFindConvolutionForwardAlgorithmEx", "cudnnHandle_t", "const cudnnTensorDescriptor_t", "const void *",
            "const cudnnFilterDescriptor_t", "const void *", "const cudnnConvolutionDescriptor_t",
            "const cudnnTensorDescriptor_t", "void *", "const int", "int *", "cudnnConvolutionFwdAlgoPerf_t *",
            "void *", "size_t"],
        ["cudnnGetConvolutionForwardAlgorithm", "cudnnHandle_t", "const cudnnTensorDescriptor_t",
            "const cudnnFilterDescriptor_t", "const cudnnConvolutionDescriptor_t", "const cudnnTensorDescriptor_t",
            "cudnnConvolutionFwdPreference_t", "size_t", "cudnnConvolutionFwdAlgo_t *"],
        ["cudnnGetConvolutionForwardWorkspaceSize", "cudnnHandle_t", "const cudnnTensorDescriptor_t",
            "const cudnnFilterDescriptor_t", "const cudnnConvolutionDescriptor_t", "const cudnnTensorDescriptor_t",
            "cudnnConvolutionFwdAlgo_t", "size_t *"],
        ["cudnnConvolutionForward", "cudnnHandle_t", "const void *", "const cudnnTensorDescriptor_t", "const void *",
            "const cudnnFilterDescriptor_t", "const void *", "const cudnnConvolutionDescriptor_t",
            "cudnnConvolutionFwdAlgo_t", "void *", "size_t", "const void *", "const cudnnTensorDescriptor_t",
            "void *"],
        ["cudnnConvolutionBackwardBias", "cudnnHandle_t", "const void *", "const cudnnTensorDescriptor_t",
            "const void *", "const void *", "const cudnnTensorDescriptor_t", "void *"],
        

        ["cudnnFindConvolutionBackwardFilterAlgorithm", "cudnnHandle_t", "const cudnnTensorDescriptor_t",
            "const cudnnTensorDescriptor_t", "const cudnnConvolutionDescriptor_t", "const cudnnFilterDescriptor_t",
            "const int", "int *", "cudnnConvolutionBwdFilterAlgoPerf_t *"],
        ["cudnnFindConvolutionBackwardFilterAlgorithmEx", "cudnnHandle_t", "const cudnnTensorDescriptor_t",
            "const void *", "const cudnnTensorDescriptor_t", "const void *", "const cudnnConvolutionDescriptor_t",
            "const cudnnFilterDescriptor_t", "void *", "const int", "int *", "cudnnConvolutionBwdFilterAlgoPerf_t *",
            "void *", "size_t"],
        ["cudnnGetConvolutionBackwardFilterAlgorithm", "cudnnHandle_t", "const cudnnTensorDescriptor_t",
            "const cudnnTensorDescriptor_t", "const cudnnConvolutionDescriptor_t", "const cudnnFilterDescriptor_t",
            "cudnnConvolutionBwdFilterPreference_t", "size_t", "cudnnConvolutionBwdFilterAlgo_t *"],
        ["cudnnGetConvolutionBackwardFilterWorkspaceSize", "cudnnHandle_t", "const cudnnTensorDescriptor_t",
            "const cudnnTensorDescriptor_t", "const cudnnConvolutionDescriptor_t", "const cudnnFilterDescriptor_t",
            "cudnnConvolutionBwdFilterAlgo_t", "size_t *"],
        ["cudnnConvolutionBackwardFilter", "cudnnHandle_t", "const void *", "const cudnnTensorDescriptor_t",
            "const void *", "const cudnnTensorDescriptor_t", "const void *", "const cudnnConvolutionDescriptor_t",
            "cudnnConvolutionBwdFilterAlgo_t", "void *", "size_t", "const void *", "const cudnnFilterDescriptor_t",
            "void *"],
        

        ["cudnnFindConvolutionBackwardDataAlgorithm", "cudnnHandle_t", "const cudnnFilterDescriptor_t",
            "const cudnnTensorDescriptor_t", "const cudnnConvolutionDescriptor_t", "const cudnnTensorDescriptor_t",
            "const int", "int *", "cudnnConvolutionBwdDataAlgoPerf_t *"],
        ["cudnnFindConvolutionBackwardDataAlgorithmEx", "cudnnHandle_t", "const cudnnFilterDescriptor_t",
            "const void *", "const cudnnTensorDescriptor_t", "const void *", "const cudnnConvolutionDescriptor_t",
            "const cudnnTensorDescriptor_t", "void *", "const int", "int *", "cudnnConvolutionBwdDataAlgoPerf_t *",
            "void *", "size_t"],
        ["cudnnGetConvolutionBackwardDataAlgorithm", "cudnnHandle_t", "const cudnnFilterDescriptor_t",
            "const cudnnTensorDescriptor_t", "const cudnnConvolutionDescriptor_t", "const cudnnTensorDescriptor_t",
            "cudnnConvolutionBwdDataPreference_t", "size_t", "cudnnConvolutionBwdDataAlgo_t *"],
        ["cudnnGetConvolutionBackwardDataWorkspaceSize", "cudnnHandle_t", "const cudnnFilterDescriptor_t",
            "const cudnnTensorDescriptor_t", "const cudnnConvolutionDescriptor_t", "const cudnnTensorDescriptor_t",
            "cudnnConvolutionBwdDataAlgo_t", "size_t *"],
        ["cudnnConvolutionBackwardData", "cudnnHandle_t", "const void *", "const cudnnFilterDescriptor_t",
            "const void *", "const cudnnTensorDescriptor_t", "const void *", "const cudnnConvolutionDescriptor_t",
            "cudnnConvolutionBwdDataAlgo_t", "void *", "size_t", "const void *", "const cudnnTensorDescriptor_t",
            "void *"],
        ["cudnnIm2Col", "cudnnHandle_t", "const cudnnTensorDescriptor_t", "const void *",
            "const cudnnFilterDescriptor_t", "const cudnnConvolutionDescriptor_t", "void *"],
        

        ["cudnnSoftmaxForward", "cudnnHandle_t", "cudnnSoftmaxAlgorithm_t", "cudnnSoftmaxMode_t", "const void *",
            "const cudnnTensorDescriptor_t", "const void *", "const void *", "const cudnnTensorDescriptor_t",
            "void *"],
        ["cudnnSoftmaxBackward", "cudnnHandle_t", "cudnnSoftmaxAlgorithm_t", "cudnnSoftmaxMode_t", "const void *",
            "const cudnnTensorDescriptor_t", "const void *", "const cudnnTensorDescriptor_t", "const void *",
            "const void *", "const cudnnTensorDescriptor_t", "void *"],
        

        ["cudnnCreatePoolingDescriptor", "cudnnPoolingDescriptor_t *"],
        ["cudnnSetPooling2dDescriptor", "cudnnPoolingDescriptor_t", "cudnnPoolingMode_t", "cudnnNanPropagation_t",
            "int", "int", "int", "int", "int", "int"],
        ["cudnnGetPooling2dDescriptor", "const cudnnPoolingDescriptor_t", "cudnnPoolingMode_t *",
            "cudnnNanPropagation_t *", "int *", "int *", "int *", "int *", "int *", "int *"],
        ["cudnnSetPoolingNdDescriptor", "cudnnPoolingDescriptor_t", "const cudnnPoolingMode_t",
            "const cudnnNanPropagation_t", "int", "const int[]", "const int[]", "const int[]"],
        ["cudnnGetPoolingNdDescriptor", "cudnnPoolingDescriptor_t", "int", "cudnnPoolingMode_t *",
            "cudnnNanPropagation_t *", "int *", "int[]", "int[]", "int[]"],
        ["cudnnGetPoolingNdForwardOutputDim", "const cudnnPoolingDescriptor_t", "const cudnnTensorDescriptor_t",
            "int", "int[]"],
        ["cudnnGetPooling2dForwardOutputDim", "const cudnnPoolingDescriptor_t", "const cudnnTensorDescriptor_t",
            "int *", "int *", "int *", "int *"],
        ["cudnnDestroyPoolingDescriptor", "cudnnPoolingDescriptor_t"],
        ["cudnnPoolingForward", "cudnnHandle_t", "const cudnnPoolingDescriptor_t", "const void *",
            "const cudnnTensorDescriptor_t", "const void *", "const void *", "const cudnnTensorDescriptor_t",
            "void *"],
        ["cudnnPoolingBackward", "cudnnHandle_t", "const cudnnPoolingDescriptor_t", "const void *",
            "const cudnnTensorDescriptor_t", "const void *", "const cudnnTensorDescriptor_t", "const void *",
            "const cudnnTensorDescriptor_t", "const void *", "const void *", "const cudnnTensorDescriptor_t",
            "void *"],
        

        ["cudnnCreateActivationDescriptor", "cudnnActivationDescriptor_t *"],
        ["cudnnSetActivationDescriptor", "cudnnActivationDescriptor_t", "cudnnActivationMode_t",
            "cudnnNanPropagation_t", "double"],
        ["cudnnGetActivationDescriptor", "const cudnnActivationDescriptor_t", "cudnnActivationMode_t *",
            "cudnnNanPropagation_t ", "double *"],
        ["cudnnDestroyActivationDescriptor", "cudnnActivationDescriptor_t"],
        ["cudnnActivationForward", "cudnnHandle_t", "cudnnActivationDescriptor_t", "const void *",
            "const cudnnTensorDescriptor_t", "const void *", "const void *", "const cudnnTensorDescriptor_t",
            "void *"],
        ["cudnnActivationBackward", "cudnnHandle_t", "cudnnActivationDescriptor_t", "const void *",
            "const cudnnTensorDescriptor_t", "const void *", "const cudnnTensorDescriptor_t", "const void *",
            "const cudnnTensorDescriptor_t", "const void *", "const void *", "const cudnnTensorDescriptor_t",
            "void *"],
        

        ["cudnnCreateLRNDescriptor", "cudnnLRNDescriptor_t *"],
        ["cudnnSetLRNDescriptor", "cudnnLRNDescriptor_t", "uint", "double", "double", "double"],
        ["cudnnGetLRNDescriptor", "cudnnLRNDescriptor_t", "uint *", "double *", "double *", "double *"],
        ["cudnnDestroyLRNDescriptor", "cudnnLRNDescriptor_t"],
        ["cudnnLRNCrossChannelForward", "cudnnHandle_t", "cudnnLRNDescriptor_t", "cudnnLRNMode_t", "const void *",
            "const cudnnTensorDescriptor_t", "const void *", "const void *", "const cudnnTensorDescriptor_t",
            "void *"],
        ["cudnnLRNCrossChannelBackward", "cudnnHandle_t", "cudnnLRNDescriptor_t", "cudnnLRNMode_t", "const void *",
            "const cudnnTensorDescriptor_t", "const void *", "const cudnnTensorDescriptor_t", "const void *",
            "const cudnnTensorDescriptor_t", "const void *", "const void *", "const cudnnTensorDescriptor_t",
            "void *"],
        ["cudnnDivisiveNormalizationForward", "cudnnHandle_t", "cudnnLRNDescriptor_t", "cudnnDivNormMode_t",
            "const void *", "const cudnnTensorDescriptor_t", "const void *", "const void *", "void *", "void *",
            "const void *", "const cudnnTensorDescriptor_t", "void *"],
        ["cudnnDivisiveNormalizationBackward", "cudnnHandle_t", "cudnnLRNDescriptor_t", "cudnnDivNormMode_t",
            "const void *", "const cudnnTensorDescriptor_t", "const void *", "const void *", "const void *",
            "void *", "void *", "const void *", "const cudnnTensorDescriptor_t", "void *", "void *"],
        

        ["cudnnDeriveBNTensorDescriptor", "cudnnTensorDescriptor_t", "const cudnnTensorDescriptor_t",
            "cudnnBatchNormMode_t"],
        ["cudnnBatchNormalizationForwardTraining", "cudnnHandle_t", "cudnnBatchNormMode_t", "const void *",
            "const void *", "const cudnnTensorDescriptor_t", "const void *", "const cudnnTensorDescriptor_t",
            "void *", "const cudnnTensorDescriptor_t", "const void *", "const void *", "double", "void *", "void *",
            "double", "void *", "void *"],
        ["cudnnBatchNormalizationForwardInference", "cudnnHandle_t", "cudnnBatchNormMode_t", "const void *",
            "const void *", "const cudnnTensorDescriptor_t", "const void *", "const cudnnTensorDescriptor_t", "void *",
            "const cudnnTensorDescriptor_t", "const void *", "const void *", "const void *", "const void *", "double"],
        ["cudnnBatchNormalizationBackward", "cudnnHandle_t", "cudnnBatchNormMode_t", "const void *", "const void *",
            "const void *", "const void *", "const cudnnTensorDescriptor_t", "const void *",
            "const cudnnTensorDescriptor_t", "const void *", "const cudnnTensorDescriptor_t", "void *",
            "const cudnnTensorDescriptor_t", "const void *", "void *", "void *", "double", "const void *",
            "const void *"],
        

        ["cudnnCreateSpatialTransformerDescriptor", "cudnnSpatialTransformerDescriptor_t *"],
        ["cudnnSetSpatialTransformerNdDescriptor", "cudnnSpatialTransformerDescriptor_t", "cudnnSamplerType_t",
            "cudnnDataType_t", "const int", "const int[]"],
        ["cudnnDestroySpatialTransformerDescriptor", "cudnnSpatialTransformerDescriptor_t"],
        ["cudnnSpatialTfGridGeneratorForward", "cudnnHandle_t", "const cudnnSpatialTransformerDescriptor_t",
            "const void *", "void *"],
        ["cudnnSpatialTfGridGeneratorBackward", "cudnnHandle_t", "const cudnnSpatialTransformerDescriptor_t",
            "const void *", "void *"],
        ["cudnnSpatialTfSamplerForward", "cudnnHandle_t", "cudnnSpatialTransformerDescriptor_t", "const void *",
            "const cudnnTensorDescriptor_t", "const void *", "const void *", "const void *", "const void *",
            "cudnnTensorDescriptor_t", "void *"],
        ["cudnnSpatialTfSamplerBackward", "cudnnHandle_t", "cudnnSpatialTransformerDescriptor_t", "const void *",
            "const cudnnTensorDescriptor_t", "const void *", "const void *", "const cudnnTensorDescriptor_t", "void *",
            "const void *", "const cudnnTensorDescriptor_t", "const void *", "const void *", "const void *", "void *"],
        

        ["cudnnCreateDropoutDescriptor", "cudnnDropoutDescriptor_t *"],
        ["cudnnDestroyDropoutDescriptor", "cudnnDropoutDescriptor_t"],
        ["cudnnDropoutGetStatesSize", "cudnnHandle_t", "size_t *"],
        ["cudnnDropoutGetReserveSpaceSize", "cudnnTensorDescriptor_t", "size_t *"],
        ["cudnnSetDropoutDescriptor", "cudnnDropoutDescriptor_t", "cudnnHandle_t", "float", "void *", "size_t",
            "ulong"],
        ["cudnnDropoutForward", "cudnnHandle_t", "const cudnnDropoutDescriptor_t", "const cudnnTensorDescriptor_t",
            "const void *", "const cudnnTensorDescriptor_t", "void *", "void *", "size_t"],
        ["cudnnDropoutBackward", "cudnnHandle_t", "const cudnnDropoutDescriptor_t", "const cudnnTensorDescriptor_t",
            "const void *", "const cudnnTensorDescriptor_t", "void *", "void *", "size_t"],
        

        ["cudnnCreateRNNDescriptor", "cudnnRNNDescriptor_t *"],
        ["cudnnDestroyRNNDescriptor", "cudnnRNNDescriptor_t"],
        ["cudnnSetRNNDescriptor", "cudnnRNNDescriptor_t", "int", "int", "cudnnDropoutDescriptor_t",
            "cudnnRNNInputMode_t", "cudnnDirectionMode_t", "cudnnRNNMode_t", "cudnnDataType_t"],
        ["cudnnGetRNNWorkspaceSize", "cudnnHandle_t", "const cudnnRNNDescriptor_t", "const int",
            "const cudnnTensorDescriptor_t *", "size_t *"],
        ["cudnnGetRNNTrainingReserveSize", "cudnnHandle_t", "const cudnnRNNDescriptor_t", "const int",
            "const cudnnTensorDescriptor_t *", "size_t *"],
        ["cudnnGetRNNParamsSize", "cudnnHandle_t", "const cudnnRNNDescriptor_t", "const cudnnTensorDescriptor_t",
            "size_t *", "cudnnDataType_t"],
        ["cudnnGetRNNLinLayerMatrixParams", "cudnnHandle_t", "const cudnnRNNDescriptor_t", "const int",
            "const cudnnTensorDescriptor_t", "const cudnnFilterDescriptor_t", "const void *", "const int",
            "cudnnFilterDescriptor_t", "void **"],
        ["cudnnGetRNNLinLayerBiasParams", "cudnnHandle_t", "const cudnnRNNDescriptor_t", "const int",
            "const cudnnTensorDescriptor_t", "const cudnnFilterDescriptor_t", "const void *", "const int",
            "cudnnFilterDescriptor_t", "void **"],
        ["cudnnRNNForwardInference", "cudnnHandle_t", "const cudnnRNNDescriptor_t", "const int",
            "const cudnnTensorDescriptor_t *", "const void *", "const cudnnTensorDescriptor_t", "const void *",
            "const void *", "const cudnnTensorDescriptor_t", "const void *", "const cudnnFilterDescriptor_t",
            "const void *", "const cudnnTensorDescriptor_t *", "void *", "const cudnnTensorDescriptor_t", "void *",
            "const cudnnTensorDescriptor_t", "void *", "void *", "size_t"],
        ["cudnnRNNForwardTraining", "cudnnHandle_t", "const cudnnRNNDescriptor_t", "const int",
            "const cudnnTensorDescriptor_t *", "const void *", "const cudnnTensorDescriptor_t", "const void *",
            "const cudnnTensorDescriptor_t", "const void *", "const cudnnFilterDescriptor_t", "const void *",
            "const cudnnTensorDescriptor_t *", "void *", "const cudnnTensorDescriptor_t", "void *",
            "const cudnnTensorDescriptor_t", "void *", "void *", "size_t", "void *", "size_t"],
        ["cudnnRNNBackwardData", "cudnnHandle_t", "const cudnnRNNDescriptor_t", "const int",
            "const cudnnTensorDescriptor_t *", "const void *", "const cudnnTensorDescriptor_t *", "const void *",
            "const cudnnTensorDescriptor_t", "const void *", "const cudnnTensorDescriptor_t", "const void *",
            "const cudnnFilterDescriptor_t", "const void *", "const cudnnTensorDescriptor_t", "const void *",
            "const cudnnTensorDescriptor_t", "const void *", "const cudnnTensorDescriptor_t *", "void *",
            "const cudnnTensorDescriptor_t", "void *", "const cudnnTensorDescriptor_t", "void *", "void *",
            "size_t", "const void *", "size_t"],
        ["cudnnRNNBackwardWeights", "cudnnHandle_t", "const cudnnRNNDescriptor_t", "const int",
            "const cudnnTensorDescriptor_t *", "const void *", "const cudnnTensorDescriptor_t", "const void *",
            "const cudnnTensorDescriptor_t *", "const void *", "const void *", "size_t",
            "const cudnnFilterDescriptor_t", "void *", "const void *", "size_t"],

        //New in cuDNN v6
        ["cudnnConvolutionBiasActivationForward", "cudnnHandle_t", "const void *", "const cudnnTensorDescriptor_t",
            "const void *", "const cudnnFilterDescriptor_t", "const void *", "const cudnnConvolutionDescriptor_t",
            "cudnnConvolutionFwdAlgo_t", "void *", "size_t", "const void *", "const cudnnTensorDescriptor_t",
            "const void *", "const cudnnTensorDescriptor_t", "const void *", "const cudnnActivationDescriptor_t",
            "const cudnnTensorDescriptor_t", "void *"]
    ];

    string generateFunctionAliases()
    {
        import std.algorithm : joiner;
        import std.conv : to;

        string ret;

        foreach(ft; functionTypes)
        {
            ret ~= "alias da_" ~ ft[0] ~ " = cudnnStatus_t function(" ~ ft[1 .. $].joiner(",").to!string ~ ");";
        }

        return ret;
    }

    string generateFunctionPointers()
    {
        string ret;

        foreach(ft; functionTypes)
        {
            ret ~= "da_" ~ ft[0] ~ " " ~ ft[0] ~ ";";
        }

        return ret;
    }

    string generateFunctionBinds()
    {
        string ret;

        foreach(ft; functionTypes)
        {
            ret ~= "bindFunc(cast(void**)&" ~ ft[0] ~ ", \"" ~ ft[0] ~ "\");";
        }

        return ret;
    }
}

struct cudnnContext;
alias cudnnHandle_t = cudnnContext*;

alias cudnnStatus_t = int;
enum : cudnnStatus_t
{
    CUDNN_STATUS_SUCCESS          = 0,
    CUDNN_STATUS_NOT_INITIALIZED  = 1,
    CUDNN_STATUS_ALLOC_FAILED     = 2,
    CUDNN_STATUS_BAD_PARAM        = 3,
    CUDNN_STATUS_INTERNAL_ERROR   = 4,
    CUDNN_STATUS_INVALID_VALUE    = 5,
    CUDNN_STATUS_ARCH_MISMATCH    = 6,
    CUDNN_STATUS_MAPPING_ERROR    = 7,
    CUDNN_STATUS_EXECUTION_FAILED = 8,
    CUDNN_STATUS_NOT_SUPPORTED    = 9,
    CUDNN_STATUS_LICENSE_ERROR    = 10
}

struct cudnnTensorStruct;
alias cudnnTensorDescriptor_t = cudnnTensorStruct*;

struct cudnnConvolutionStruct;
alias cudnnConvolutionDescriptor_t = cudnnConvolutionStruct*;

struct cudnnPoolingStruct;
alias cudnnPoolingDescriptor_t = cudnnPoolingStruct*;

struct cudnnFilterStruct;
alias cudnnFilterDescriptor_t = cudnnFilterStruct*;

struct cudnnLRNStruct;
alias cudnnLRNDescriptor_t = cudnnLRNStruct*;

struct cudnnActivationStruct;
alias cudnnActivationDescriptor_t = cudnnActivationStruct*;

struct cudnnSpatialTransformerStruct;
alias cudnnSpatialTransformerDescriptor_t = cudnnSpatialTransformerStruct*;

struct cudnnOpTensorStruct;
alias cudnnOpTensorDescriptor_t = cudnnOpTensorStruct*;

alias cudnnDataType_t = int;
enum : cudnnDataType_t
{
    CUDNN_DATA_FLOAT  = 0,
    CUDNN_DATA_DOUBLE = 1,
    CUDNN_DATA_HALF   = 2
}

alias cudnnNanPropagation_t = int;
enum : cudnnNanPropagation_t
{
    CUDNN_NOT_PROPAGATE_NAN  = 0,
    CUDNN_PROPAGATE_NAN      = 1
}

alias cudnnTensorFormat_t = int;
enum : cudnnTensorFormat_t
{
    CUDNN_TENSOR_NCHW = 0,   /* row major (wStride = 1, hStride = w) */
    CUDNN_TENSOR_NHWC = 1    /* feature maps interleaved ( cStride = 1 )*/
}

alias cudnnOpTensorOp_t = int;
enum : cudnnOpTensorOp_t
{
    CUDNN_OP_TENSOR_ADD = 0,
    CUDNN_OP_TENSOR_MUL = 1,
    CUDNN_OP_TENSOR_MIN = 2,
    CUDNN_OP_TENSOR_MAX = 3
}

alias cudnnConvolutionMode_t = int;
enum : cudnnConvolutionMode_t
{
    CUDNN_CONVOLUTION       = 0,
    CUDNN_CROSS_CORRELATION = 1
}

alias cudnnConvolutionFwdPreference_t = int;
enum : cudnnConvolutionFwdPreference_t
{
    CUDNN_CONVOLUTION_FWD_NO_WORKSPACE            = 0,
    CUDNN_CONVOLUTION_FWD_PREFER_FASTEST          = 1,
    CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT = 2
}

alias cudnnConvolutionFwdAlgo_t = int;
enum : cudnnConvolutionFwdAlgo_t
{
    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM         = 0,
    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM = 1,
    CUDNN_CONVOLUTION_FWD_ALGO_GEMM                  = 2,
    CUDNN_CONVOLUTION_FWD_ALGO_DIRECT                = 3,
    CUDNN_CONVOLUTION_FWD_ALGO_FFT                   = 4,
    CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING            = 5,
    CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD              = 6,
    CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED     = 7
}

struct cudnnConvolutionFwdAlgoPerf_t
{
    cudnnConvolutionFwdAlgo_t   algo;
    cudnnStatus_t               status;
    float                       time;
    size_t                      memory;
    int[5] reserved;
}

alias cudnnConvolutionBwdFilterPreference_t = int;
enum : cudnnConvolutionBwdFilterPreference_t
{
    CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE            = 0,
    CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST          = 1,
    CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT = 2
}

alias cudnnConvolutionBwdFilterAlgo_t = int;
enum : cudnnConvolutionBwdFilterAlgo_t
{
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0         = 0,  // non-deterministic
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1         = 1,
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT       = 2,
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3         = 3,  // non-deterministic, algo0 with workspace
    // CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD  = 4, // not implemented
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED = 5
}

struct cudnnConvolutionBwdFilterAlgoPerf_t
{
    cudnnConvolutionBwdFilterAlgo_t algo;
    cudnnStatus_t status;
    float time;
    size_t memory;
    int[5] reserved;
}

alias cudnnConvolutionBwdDataPreference_t = int;
enum : cudnnConvolutionBwdDataPreference_t
{
    CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE             = 0,
    CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST           = 1,
    CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT  = 2
}

alias cudnnConvolutionBwdDataAlgo_t = int;
enum : cudnnConvolutionBwdDataAlgo_t
{
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_0                 = 0, // non-deterministic
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_1                 = 1,
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT               = 2,
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING        = 3,
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD          = 4,
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED = 5
}

struct cudnnConvolutionBwdDataAlgoPerf_t
{
    cudnnConvolutionBwdDataAlgo_t   algo;
    cudnnStatus_t                   status;
    float                           time;
    size_t                          memory;
    int[5] reserved;
}

alias cudnnSoftmaxAlgorithm_t = int;
enum : cudnnSoftmaxAlgorithm_t
{
    CUDNN_SOFTMAX_FAST     = 0,         /* straightforward implementation */
    CUDNN_SOFTMAX_ACCURATE = 1,         /* subtract max from every point to avoid overflow */
    CUDNN_SOFTMAX_LOG      = 2
}

alias cudnnSoftmaxMode_t = int;
enum : cudnnSoftmaxMode_t
{
    CUDNN_SOFTMAX_MODE_INSTANCE = 0,   /* compute the softmax over all C, H, W for each N */
    CUDNN_SOFTMAX_MODE_CHANNEL = 1
}

alias cudnnPoolingMode_t = int;
enum : cudnnPoolingMode_t
{
    CUDNN_POOLING_MAX     = 0,
    CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING = 1, // count for average includes padded values
    CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING = 2
}

alias cudnnActivationMode_t = int;
enum : cudnnActivationMode_t
{
    CUDNN_ACTIVATION_SIGMOID      = 0,
    CUDNN_ACTIVATION_RELU         = 1,
    CUDNN_ACTIVATION_TANH         = 2,
    CUDNN_ACTIVATION_CLIPPED_RELU = 3
}

alias cudnnLRNMode_t = int;
enum : cudnnLRNMode_t
{
    CUDNN_LRN_CROSS_CHANNEL_DIM1 = 0
}

alias cudnnDivNormMode_t = int;
enum : cudnnDivNormMode_t
{
    CUDNN_DIVNORM_PRECOMPUTED_MEANS = 0
}

alias cudnnBatchNormMode_t = int;
enum : cudnnBatchNormMode_t
{
    // bnScale, bnBias tensor dims are 1xCxHxWx.. (one value per CHW...-slice, normalized over N slice)
    CUDNN_BATCHNORM_PER_ACTIVATION = 0,

    //bnScale, bnBias tensor dims are 1xCx1x1 (one value per C-dim normalized over Nx1xHxW subtensors)
    CUDNN_BATCHNORM_SPATIAL        = 1
}

alias cudnnSamplerType_t = int;
enum : cudnnSamplerType_t
{
    CUDNN_SAMPLER_BILINEAR=0
}

struct cudnnDropoutStruct;
alias cudnnDropoutDescriptor_t = cudnnDropoutStruct*;

alias cudnnRNNMode_t = int;
enum : cudnnRNNMode_t
{
    CUDNN_RNN_RELU = 0, // Stock RNN with ReLu activation
    CUDNN_RNN_TANH = 1, // Stock RNN with tanh activation
    CUDNN_LSTM = 2,     // LSTM with no peephole connections
    CUDNN_GRU = 3       // Using h' = tanh(r * Uh(t-1) + Wx) and h = (1 - z) * h' + z * h(t-1);
}

alias cudnnDirectionMode_t = int;
enum : cudnnDirectionMode_t
{
    CUDNN_UNIDIRECTIONAL = 0,
    CUDNN_BIDIRECTIONAL = 1      // Using output concatination at each step. Do we also want to support output sum?
}

alias cudnnRNNInputMode_t = int;
enum : cudnnRNNInputMode_t
{
    CUDNN_LINEAR_INPUT = 0,
    CUDNN_SKIP_INPUT = 1
}

struct cudnnRNNStruct;
alias cudnnRNNDescriptor_t = cudnnRNNStruct*;

extern(System) @nogc nothrow
{
    alias da_cudnnGetErrorString = const char *function(cudnnStatus_t);

    mixin(generateFunctionAliases());
}

__gshared
{
    da_cudnnGetErrorString cudnnGetErrorString;

    mixin(generateFunctionPointers());
}

class DerelictCuDNN7Loader : SharedLibLoader
{
    public
    {
        this()
        {
            super(libNames);
        }
    }

    protected
    {
        override void loadSymbols()
        {
            bindFunc(cast(void**)&cudnnGetErrorString, "cudnnGetErrorString");

            mixin(generateFunctionBinds());
        }
    }
}

__gshared DerelictCuDNN7Loader DerelictCuDNN7;

shared static this()
{
    DerelictCuDNN7 = new DerelictCuDNN7Loader();
}

version (grain_cuda) unittest
{
    import std.conv : to;
    import std.stdio : writeln;

    try
    {
        DerelictCuDNN7.load();
        writeln("Successfully loaded cuDNN v7");
    }
    catch(Exception e)
    {
        writeln("Could not load cuDNN v7");
    }
}


/* Maximum supported number of tensor dimensions */
enum CUDNN_DIM_MAX = 8;
