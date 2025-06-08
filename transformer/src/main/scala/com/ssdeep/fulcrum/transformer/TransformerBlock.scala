package com.ssdeep.fulcrum.transformer

import com.ssdeep.fulcrum.attention.MultiheadAttention
import com.ssdeep.fulcrum.transformer.config.GPTConfig
import com.ssdeep.fulcrum.transformer.nn.FeedForward
import com.ssdeep.fulcrum.transformer.normalization.LayerNormalization
import torch.nn.modules.TensorModule
import torch.{Float32, Tensor}

class TransformerBlock(cfg: GPTConfig) extends TensorModule[Float32]:
  private val attention = MultiheadAttention(
    dIn = cfg.embedDim,
    dOut = cfg.embedDim, 
    contextLength = cfg.contextLength, 
    dropoutFactor = cfg.dropRate, 
    qkvBias = cfg.qkvBias, 
    numHeads = cfg.numHeads
  )
  private val ff = FeedForward(cfg)
  private val norm1 = LayerNormalization(cfg.embedDim)
  private val norm2 = LayerNormalization(cfg.embedDim)
  private val dropShortCut = torch.nn.Dropout(cfg.dropRate)
  
  def apply(x: Tensor[Float32]): Tensor[Float32] =
    var shortcut = x
    shortcut = norm1(shortcut)
    shortcut = attention(shortcut)
    shortcut = dropShortCut(shortcut)
    shortcut = shortcut + x
    
    val y = shortcut
    
    shortcut = norm2(shortcut)
    shortcut = ff(shortcut)
    shortcut = dropShortCut(shortcut)
    shortcut = shortcut + y
    y.native.deallocate()
    shortcut
