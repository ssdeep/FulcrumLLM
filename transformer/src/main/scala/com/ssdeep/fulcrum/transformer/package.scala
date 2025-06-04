package com.ssdeep.fulcrum

import com.ssdeep.fulcrum.transformer.nn.NLayeredDeepNeuralNetwork
import torch.{Float32, Tensor, noGrad}
import org.bytedeco.pytorch
import org.bytedeco.pytorch.global.torch as torchNative
import torch.Tensor.fromNative

package object transformer {

  extension (model: NLayeredDeepNeuralNetwork) def print_gradients(x: Tensor[Float32]): Unit =
    val output = model(x)
    val target = torch.Tensor(Seq(0))
    val lossFn = fromNative(
      torchNative.mse_loss(
      x.native,
      output.native
    ))
    lossFn.backward()
    model.namedParameters(true)
      .filter(_._1.contains("weight"))
      .foreach {
        case (w, value) =>
          println(s"$w and gradient mean of ${value.grad.get.abs.mean.item}")
      }
}
