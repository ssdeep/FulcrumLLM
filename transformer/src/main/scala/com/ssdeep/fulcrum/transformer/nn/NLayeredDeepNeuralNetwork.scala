package com.ssdeep.fulcrum.transformer.nn
import com.ssdeep.fulcrum.transformer.activations.GELU
import torch.*
import torch.nn.*

class NLayeredDeepNeuralNetwork(layerSizes: Array[Int], useShortcut: Boolean) extends torch.nn.modules.TensorModule[Float32]:
  val layers = register(
    new torch.nn.ModuleList(
      (1 until layerSizes.length).map {
        i => 
          new torch.nn.Sequential(
            new Linear(layerSizes(i - 1), layerSizes(i)),
            new GELU()
          )
      }:_*
    )
  )
  
  def apply(x: Tensor[Float32]): Tensor[Float32] =
    var output = x 
    layers.foreach {
      layer => 
        val layerOutput = layer(output)
        if useShortcut then output = output + layerOutput
        else output = layerOutput
    }
    output


