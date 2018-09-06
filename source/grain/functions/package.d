/**
   A type defined in grain.functions is equivalent to chainer.Function or torch.autograd.Function.
   Function is a set of core autograd algorithms for grain.chain that compose some functions in computation graph.
 */
module grain.functions;

public import grain.functions.common;
public import grain.functions.unary;
public import grain.functions.binary;
public import grain.functions.loss;
public import grain.functions.reduction;
public import grain.functions.topology;
