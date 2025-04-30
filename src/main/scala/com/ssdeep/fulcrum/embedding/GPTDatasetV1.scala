package com.ssdeep.fulcrum.embedding

import com.ssdeep.fulcrum.tokenizer.Tokenizer

import scala.collection.mutable.ArrayBuffer
import torch.*

/**
 * Reference Dataset that takes text and returns input and target tensors
 */
case class GPTDatasetV1(text: String, tokenizer: Tokenizer, maxLength: Int, stride: Int):

  private val tokenIds: List[Long] = tokenizer.encode(text)
  private val inputChunks = torch.Tensor(tokenIds.sliding(maxLength, stride).toSeq.dropRight(1))
  private val targetChunks = torch.Tensor(tokenIds.drop(1).sliding(maxLength, stride).toSeq)

  def getDataset = torch.data.TensorDataset(inputChunks, targetChunks)