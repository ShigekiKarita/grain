/**
Metric (e.g., accuracy)

TODO: perplexity, AUC, F1, BLEU, edit distance
*/
module grain.metric;

import grain.autograd : isVariable;


/// compute accuracy comparing prediction y (histgram) to target t (id)
auto accuracy(Vy, Vt)(Vy y, Vt t) if (isVariable!Vy && isVariable!Vt) {
    import mir.ndslice : maxIndex;
    import grain.autograd : to, HostStorage;

    auto nbatch = t.shape[0];
    auto hy = y.to!HostStorage.sliced;
    auto ht = t.to!HostStorage.sliced;
    double acc = 0.0;
    foreach (i; 0 .. nbatch) {
        auto maxid = hy[i].maxIndex[0];
        if (maxid == ht[i]) {
            ++acc;
        }
    }
    return acc / nbatch;
}
