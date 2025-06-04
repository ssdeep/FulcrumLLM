package com.ssdeep.fulcrum.attention
import torch.Tensor.fromNative
import torch.{Float32, Tensor, nn}

class CompactSelfAttention(dIn: Int, dOut: Int, qkvBias: Boolean = false) extends torch.nn.Module {
   val Wquery = nn.Linear(dIn, dOut, qkvBias)
   val Wkey = nn.Linear(dIn, dOut, qkvBias)
   val Wvalue = nn.Linear(dIn, dOut, qkvBias)

   def apply(input: Tensor[Float32]): Tensor[Float32] =
     val keys = input.map(Wkey.apply)
     val values = input.map(Wvalue.apply)
     val queries = input.map(Wquery.apply)
     val attnScores = queries  % keys
     val attnWeights = torch.softmax(attnScores/Math.pow(keys.shape.last, 0.5), dim = -1L, dtype = torch.float32)
     val maskSimple = torch.tril(torch.ones(Seq(attnWeights.shape.head, attnWeights.shape.head)))
     val maskedAttention = attnWeights * maskSimple
     println(maskedAttention)
     val rowSums = maskedAttention.sum(dim = -1, keepdim = true, dtype = torch.float32)
     val maskedAttnNormalized = maskedAttention / rowSums
//     val softMaxMaskedAttn = torch.softmax()
     val contextVector = attnWeights `@` values
     contextVector
}
