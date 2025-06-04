package com.ssdeep.fulcrum.transformer.activations

import torch.{Float32, Tensor}
import torch.*
import org.bytedeco.pytorch
import torch.Tensor.fromNative
import torch.nn.modules.TensorModule

class GELU extends TensorModule[Float32]:
  def apply(x: Tensor[Float32]): Tensor[Float32] = {
    val root2byPi = Math.sqrt(2.0/Math.PI).toFloat
    val innerSum = (x + torch.pow(x, 3) * torch.Tensor((0.044715).toFloat))
    val outerSum = torch.tanh(Tensor(root2byPi) * innerSum) + 1
    x * 0.5.floatValue() * outerSum
  }
