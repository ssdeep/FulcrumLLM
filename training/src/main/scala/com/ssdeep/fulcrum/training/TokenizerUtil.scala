package com.ssdeep.fulcrum.training

import com.ssdeep.fulcrum.tokenizer.Tokenizer
import torch.{Int64, Tensor}

object TokenizerUtil:
  private val tokenizer = Tokenizer.buildDefaultO200kBase

  def textToTokenIds(text: String): Tensor[Int64] =
    torch.Tensor(tokenizer.encode(text)).unsqueeze(0)

  def tokenIdsToText(tokenIds: Tensor[Int64]): String = {
    val tokenIdsList = tokenIds.squeeze(0).toArray.toList
    tokenizer.decode(tokenIdsList)
  }
