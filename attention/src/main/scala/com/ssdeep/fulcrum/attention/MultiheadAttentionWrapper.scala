package com.ssdeep.fulcrum.attention

import torch.{Float32, Tensor}
import torch.nn.modules.TensorModule
import org.bytedeco.pytorch.global.torch as pytorch
class MultiheadAttentionWrapper(
                                 dIn: Int,
                                 dOut: Int,
                                 contextLength: Int,
                                 dropoutFactor: Double = 0.5,
                                 qkvBias: Boolean = false,
                                 numHeads: Int = 5) extends torch.nn.Module:
  val causalAttentionHeads = (0 until 5).map(_ => new CausalAttention(
    dIn, dOut, contextLength, dropoutFactor, qkvBias
  ))

  def apply(input: Tensor[Float32]): Tensor[Float32] =
    torch.cat(
      causalAttentionHeads.map(p => p.apply(input)),
      dim = -1
    )


