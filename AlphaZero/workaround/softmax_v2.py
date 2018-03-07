from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.eager import context

def _ensure_xent_args(name, sentinel, labels, logits):
  # Make sure that all arguments were passed as named arguments.
  if sentinel is not None:
    raise ValueError("Only call `%s` with "
                     "named arguments (labels=..., logits=..., ...)" % name)
  if labels is None or logits is None:
    raise ValueError("Both labels and logits must be provided.")

def _flatten_outer_dims(logits):
  """Flattens logits' outer dimensions and keep its last dimension."""
  rank = array_ops.rank(logits)
  last_dim_size = array_ops.slice(
      array_ops.shape(logits), [math_ops.subtract(rank, 1)], [1])
  output = array_ops.reshape(logits, array_ops.concat([[-1], last_dim_size], 0))

  # Set output shape if known.
  if context.in_graph_mode():
    shape = logits.get_shape()
    if shape is not None and shape.dims is not None:
      shape = shape.as_list()
      product = 1
      product_valid = True
      for d in shape[:-1]:
        if d is None:
          product_valid = False
          break
        else:
          product *= d
      if product_valid:
        output_shape = [product, shape[-1]]
        output.set_shape(output_shape)

  return output

def softmax_cross_entropy_with_logits_v2(_sentinel=None,  # pylint: disable=invalid-name
                                         labels=None, logits=None,
                                         dim=-1, name=None):
  """Computes softmax cross entropy between `logits` and `labels`.

  Measures the probability error in discrete classification tasks in which the
  classes are mutually exclusive (each entry is in exactly one class).  For
  example, each CIFAR-10 image is labeled with one and only one label: an image
  can be a dog or a truck, but not both.

  **NOTE:**  While the classes are mutually exclusive, their probabilities
  need not be.  All that is required is that each row of `labels` is
  a valid probability distribution.  If they are not, the computation of the
  gradient will be incorrect.

  If using exclusive `labels` (wherein one and only
  one class is true at a time), see `sparse_softmax_cross_entropy_with_logits`.

  **WARNING:** This op expects unscaled logits, since it performs a `softmax`
  on `logits` internally for efficiency.  Do not call this op with the
  output of `softmax`, as it will produce incorrect results.

  `logits` and `labels` must have the same shape, e.g.
  `[batch_size, num_classes]` and the same dtype (either `float16`, `float32`,
  or `float64`).

  Backpropagation will happen into both `logits` and `labels`.  To disallow
  backpropagation into `labels`, pass label tensors through a `stop_gradients`
  before feeding it to this function.

  **Note that to avoid confusion, it is required to pass only named arguments to
  this function.**

  Args:
    _sentinel: Used to prevent positional parameters. Internal, do not use.
    labels: Each row `labels[i]` must be a valid probability distribution.
    logits: Unscaled log probabilities.
    dim: The class dimension. Defaulted to -1 which is the last dimension.
    name: A name for the operation (optional).

  Returns:
    A 1-D `Tensor` of length `batch_size` of the same type as `logits` with the
    softmax cross entropy loss.
  """
  _ensure_xent_args("softmax_cross_entropy_with_logits", _sentinel,
                    labels, logits)

  # TODO(pcmurray) Raise an error when the labels do not sum to 1. Note: This
  # could break users who call this with bad labels, but disregard the bad
  # results.

  with ops.name_scope(
      name, "softmax_cross_entropy_with_logits", [logits, labels]) as name:
    logits = ops.convert_to_tensor(logits, name="logits")
    labels = ops.convert_to_tensor(labels, name="labels")
    precise_logits = math_ops.cast(logits, dtypes.float32) if (
        logits.dtype == dtypes.float16) else logits
    # labels and logits must be of the same type
    labels = math_ops.cast(labels, precise_logits.dtype)
    input_rank = array_ops.rank(precise_logits)
    # For shape inference.
    shape = logits.get_shape()

    # Move the dim to the end if dim is not the last dimension.
    if dim is not -1:
      def _move_dim_to_end(tensor, dim_index, rank):
        return array_ops.transpose(tensor,
                                   array_ops.concat([
                                       math_ops.range(dim_index),
                                       math_ops.range(dim_index + 1, rank),
                                       [dim_index]
                                   ], 0))

      precise_logits = _move_dim_to_end(precise_logits, dim, input_rank)
      labels = _move_dim_to_end(labels, dim, input_rank)

    input_shape = array_ops.shape(precise_logits)

    # Make precise_logits and labels into matrices.
    precise_logits = _flatten_outer_dims(precise_logits)
    labels = _flatten_outer_dims(labels)

    # Do the actual op computation.
    # The second output tensor contains the gradients.  We use it in
    # _CrossEntropyGrad() in nn_grad but not here.
    cost, unused_backprop = gen_nn_ops._softmax_cross_entropy_with_logits(
        precise_logits, labels, name=name)

    # The output cost shape should be the input minus dim.
    output_shape = array_ops.slice(input_shape, [0],
                                   [math_ops.subtract(input_rank, 1)])
    cost = array_ops.reshape(cost, output_shape)

    # Make shape inference work since reshape and transpose may erase its static
    # shape.
    if context.in_graph_mode() and shape is not None and shape.dims is not None:
      shape = shape.as_list()
      del shape[dim]
      cost.set_shape(shape)

    if logits.dtype == dtypes.float16:
      return math_ops.cast(cost, dtypes.float16)
    else:
      return cost


_XENT_DEPRECATION = """
Future major versions of TensorFlow will allow gradients to flow
into the labels input on backprop by default.

See tf.nn.softmax_cross_entropy_with_logits_v2.
"""
