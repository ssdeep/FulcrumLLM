package com.ssdeep.fulcrum.transformer.nn

import com.ssdeep.fulcrum.transformer.activations.GELU
import com.ssdeep.fulcrum.transformer.config.GPTConfig
import torch.{Float32, Tensor}

case class FeedForward(config: GPTConfig) extends torch.nn.Module:
  
  val layers = register(new torch.nn.Sequential(
    new torch.nn.Linear(config.embedDim, config.embedDim * 4),
    new GELU(),
    new torch.nn.Linear(config.embedDim * 4, config.embedDim)
   )
  )
  
  def apply(x: Tensor[Float32]) =
    layers(x)
