package com.ssdeep.fulcrum.transformer.normalization

import torch.{Float32, Tensor}

case class LayerNormalization(embedDim: Int) extends torch.nn.Module:
  private val eps = (1e-5).toFloat

  private val scale = registerParameter(torch.ones(embedDim))
  private val shift = registerParameter(torch.zeros(embedDim))
  
  def apply(input: Tensor[Float32]): Tensor[Float32] =
    val mean = input.mean(dim = -1, keepdim = true)
    val variance: Tensor[Float32] = torch.variance[Float32](input, dim = -1, keepdim = true)
    val normalized = (input - mean)/torch.sqrt[Float32](variance + eps)
    variance.native.deallocate()
    mean.native.deallocate()
    scale * normalized + shift
