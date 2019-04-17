module grain.warpctc;

extern (C):

//forward declare of CUDA typedef to avoid needing to pull in CUDA headers
struct CUstream_st;
alias CUstream = CUstream_st*;

alias ctcStatus_t = int;
enum : ctcStatus_t
{
    CTC_STATUS_SUCCESS = 0,
    CTC_STATUS_MEMOPS_FAILED = 1,
    CTC_STATUS_INVALID_VALUE = 2,
    CTC_STATUS_EXECUTION_FAILED = 3,
    CTC_STATUS_UNKNOWN_ERROR = 4
}

/** Returns a single integer which specifies the API version of the warpctc library */
int get_warpctc_version();

/**
Returns a string containing a description of status that was passed in

Params:
    status = identifies which string should be returned

Returns: C style string containing the text description
 */
const(char)* ctcGetStatusString(ctcStatus_t status);

alias ctcComputeLocation = int;
enum : ctcComputeLocation
{
    CTC_CPU = 0,
    CTC_GPU = 1
}

/**
Structure used for options to the CTC compution.

Applications should zero out the array using memset and sizeof(struct
 ctcOptions) in C or default initialization (e.g. 'ctcOptions
options{};' or 'auto options = ctcOptions{}') in C++ to ensure
forward compatibility with added options. */
struct ctcOptions
{
    /// indicates where the ctc calculation should take place {CTC_CPU | CTC_GPU}
    ctcComputeLocation loc;
    union
    {
        /// used when loc == CTC_CPU, the maximum number of threads that can be used
        uint num_threads;

        /// used when loc == CTC_GPU, which stream the kernels should be launched in
        CUstream stream;
    };

    /// the label value/index that the CTC calculation should use as the blank label
    int blank_label;
};



/** Compute the connectionist temporal classification loss between a sequence
 *  of probabilities and a ground truth labeling.  Optionally compute the
 *  gradient with respect to the inputs.
 * \param [in] activations pointer to the activations in either CPU or GPU
 *             addressable memory, depending on info.  We assume a fixed
 *             memory layout for this 3 dimensional tensor, which has dimension
 *             (t, n, p), where t is the time index, n is the minibatch index,
 *             and p indexes over probabilities of each symbol in the alphabet.
 *             The memory layout is (t, n, p) in C order (slowest to fastest changing
 *             index, aka row-major), or (p, n, t) in Fortran order (fastest to slowest
 *             changing index, aka column-major). We also assume strides are equal to
 *             dimensions - there is no padding between dimensions.
 *             More precisely, element (t, n, p), for a problem with mini_batch examples
 *             in the mini batch, and alphabet_size symbols in the alphabet, is located at:
 *             activations[(t * mini_batch + n) * alphabet_size + p]
 * \param [out] gradients if not NULL, then gradients are computed.  Should be
 *              allocated in the same memory space as probs and memory
 *              ordering is identical.
 * \param [in]  flat_labels Always in CPU memory.  A concatenation
 *              of all the labels for the minibatch.
 * \param [in]  label_lengths Always in CPU memory. The length of each label
 *              for each example in the minibatch.
 * \param [in]  input_lengths Always in CPU memory.  The number of time steps
 *              for each sequence in the minibatch.
 * \param [in]  alphabet_size The number of possible output symbols.  There
 *              should be this many probabilities for each time step.
 * \param [in]  mini_batch How many examples in a minibatch.
 * \param [out] costs Always in CPU memory.  The cost of each example in the
 *              minibatch.
 * \param [in,out] workspace In same memory space as probs. Should be of
 *                 size requested by get_workspace_size.
 * \param [in]  options see struct ctcOptions
 *
 *  \return Status information
 *
 * */
ctcStatus_t compute_ctc_loss(const(float)* activations,
                             float* gradients,
                             const(int)* flat_labels,
                             const(int)* label_lengths,
                             const(int)* input_lengths,
                             int alphabet_size,
                             int minibatch,
                             float *costs,
                             void *workspace,
                             ctcOptions options);


/** For a given set of labels and minibatch size return the required workspace
 *  size.  This will need to be allocated in the same memory space as your
 *  probabilities.
 * \param [in]  label_lengths Always in CPU memory. The length of each label
 *              for each example in the minibatch.
 * \param [in]  input_lengths Always in CPU memory.  The number of time steps
 *              for each sequence in the minibatch.
 * \param [in]  alphabet_size How many symbols in the alphabet or, equivalently,
 *              the number of probabilities at each time step
 * \param [in]  mini_batch How many examples in a minibatch.
 * \param [in]  info see struct ctcOptions
 * \param [out] size_bytes is pointer to a scalar where the memory
 *              requirement in bytes will be placed. This memory should be allocated
 *              at the same place, CPU or GPU, that the probs are in
 *
 *  \return Status information
 **/
ctcStatus_t get_workspace_size(const(int)* label_lengths,
                               const(int)* input_lengths,
                               int alphabet_size, int minibatch,
                               ctcOptions info,
                               size_t* size_bytes);


// Numerically stable softmax for a minibatch of 1
private void softmax(const(float)* acts, int alphabet_size, int T, float *probs)
{
    import std.math;
    for (int t = 0; t < T; ++t)
    {
        float max_activation = -float.infinity;
        for (int a = 0; a < alphabet_size; ++a)
            max_activation = fmax(max_activation, acts[t*alphabet_size + a]);

        float denom = 0;
        for (int a = 0; a < alphabet_size; ++a)
            denom += exp(acts[t*alphabet_size + a] - max_activation);

        for (int a = 0; a < alphabet_size; ++a)
            probs[t*alphabet_size + a] = exp(acts[t*alphabet_size + a] - max_activation) / denom;
    }
}


void throw_on_error(ctcStatus_t status, string message)
{
    import std.format : format;
    import std.string : fromStringz;

    assert(status == CTC_STATUS_SUCCESS,
           format!"%s, stat = %s"(message, ctcGetStatusString(status).fromStringz));
}


unittest
{
    import std.stdio;
    import std.math;
    import core.stdc.stdlib;
    assert(get_warpctc_version() == 2);
    // https://github.com/baidu-research/warp-ctc/blob/master/tests/test_cpu.cpp

    const int alphabet_size = 5;
    const int T = 2;

    float[] activations = [0.1, 0.6, 0.1, 0.1, 0.1,
                           0.1, 0.1, 0.6, 0.1, 0.1];

    // Calculate the score analytically
    float expected_score;
    {
        auto probs = new float[activations.length];
        softmax(activations.ptr, alphabet_size, T, probs.ptr);

        // Score calculation is specific to the given activations above
        expected_score = probs[1] * probs[7];
    }

    int[] labels = [1, 2];
    int[] label_lengths = [2];
    int[] lengths = [T];

    float score;
    ctcOptions options;
    options.loc = CTC_CPU;
    options.num_threads = 1;

    size_t cpu_alloc_bytes;
    throw_on_error(get_workspace_size(label_lengths.ptr, lengths.ptr,
                                      alphabet_size, cast(int) lengths.length, options,
                                      &cpu_alloc_bytes),
                   "Error: get_workspace_size in small_test");

    void* ctc_cpu_workspace = malloc(cpu_alloc_bytes);

    throw_on_error(compute_ctc_loss(activations.ptr, null,
                                    labels.ptr, label_lengths.ptr,
                                    lengths.ptr,
                                    alphabet_size,
                                    cast(int) lengths.length,
                                    &score,
                                    ctc_cpu_workspace,
                                    options),
                   "Error: compute_ctc_loss in small_test");

    free(ctc_cpu_workspace);
    score = exp(-score);
    const float eps = 1e-6;

    const float lb = expected_score - eps;
    const float ub = expected_score + eps;
    assert(score > lb && score < ub);
}

version (grain_cuda) unittest
{
    import derelict.cuda;
    import grain.cuda;
    import std.math;

    const int alphabet_size = 5;
    const int T = 2;

    float[] activations = [0.1, 0.6, 0.1, 0.1, 0.1,
                           0.1, 0.1, 0.6, 0.1, 0.1];

    // Calculate the score analytically
    float expected_score;
    {
        auto probs = new float[activations.length];
        softmax(activations.ptr, alphabet_size, T, probs.ptr);

        // Score calculation is specific to the given activations above
        expected_score = probs[1] * probs[7];
    }

    CUstream stream;
    cuStreamCreate(cast(void**) &stream, CU_STREAM_DEFAULT);
    scope (exit) cuStreamDestroy(stream);
    // throw_on_error(cudaStreamCreate(&stream),
    //                "cudaStreamCreate");

    // float *activations_gpu;
    auto activations_gpu = CuPtr!float(activations);
    // throw_on_error(cudaMalloc(cast(void**) &activations_gpu,
    //                activations.length * float.sizeof),
    //                "cudaMalloc");
    // throw_on_error(cudaMemcpyAsync(activations_gpu, activations.ptr,
    //                                activations.length * float.sizeof,
    //                                cudaMemcpyHostToDevice, stream),
    //                "cudaMemcpyAsync");

    int[] labels = [1, 2];
    int[] label_lengths = [2];
    int[] lengths = [T];

    float score;

    ctcOptions options;
    options.loc = CTC_GPU;
    options.stream = cast(CUstream) stream;

    size_t gpu_alloc_bytes;
    throw_on_error(get_workspace_size(label_lengths.ptr, lengths.ptr,
                                      alphabet_size, cast(int) lengths.length, options,
                                      &gpu_alloc_bytes),
                   "Error: get_workspace_size in small_test");

    // char *ctc_gpu_workspace;
    auto ctc_gpu_workspace = CuPtr!char(gpu_alloc_bytes);
    // throw_on_error(cudaMalloc(cast(void**) &ctc_gpu_workspace, gpu_alloc_bytes),
    //                "cudaMalloc");

    throw_on_error(compute_ctc_loss(activations_gpu.data, null,
                                    labels.ptr, label_lengths.ptr,
                                    lengths.ptr,
                                    alphabet_size,
                                    cast(int) lengths.length,
                                    &score,
                                    ctc_gpu_workspace.data,
                                    options),
                   "Error: compute_ctc_loss in small_test");

    score = exp(-score);
    const float eps = 1e-6;

    const float lb = expected_score - eps;
    const float ub = expected_score + eps;

    // throw_on_error(cudaFree(activations_gpu),
    //                "cudaFree");
    // throw_on_error(cudaFree(ctc_gpu_workspace),
    //                "cudaFree");
    // throw_on_error(cudaStreamDestroy(stream),
    //                "cudaStreamDestroy");

    assert(score > lb && score < ub);
}
