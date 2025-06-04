import com.ssdeep.fulcrum.attention.{CausalAttention, CompactSelfAttention, MultiheadAttention, MultiheadAttentionWrapper, SimpleAttention, WeightedAttention}
import com.ssdeep.fulcrum.core.ClassNameLogger
import com.ssdeep.fulcrum.embedding.GPTDatasetV1
import com.ssdeep.fulcrum.tokenizer.Tokenizer
import org.bytedeco.pytorch.MultiheadAttentionImpl
import torch.{Float32, Tensor}

import scala.io.Source

object MultiheadAttentionTest extends ClassNameLogger:
  def main(args: Array[String]): Unit =
    val tokenizer = Tokenizer.buildDefaultO200kBase
    val textFromVerdict = Source.fromResource("The_Verdict.txt")
      .getLines().mkString("")
    val MAX_LENGTH = 5L
    val BATCH_SIZE = 8
    val gptDataset = GPTDatasetV1(
      textFromVerdict,
      tokenizer = tokenizer,
      maxLength = MAX_LENGTH.intValue(),
      stride = 1,
      batchSize = BATCH_SIZE
    )

    torch.manualSeed(128L)

//    val simpleAttention = WeightedAttention(gptDataset)
    val (batch, target) = gptDataset.dataLoader.iterator.next()
    val batchEmbeddings = gptDataset.inputEmbeddings(batch)
    val causalAttention = new CausalAttention(batchEmbeddings.shape.last, 100, 5)
    val multiheadAttention = new MultiheadAttention(batchEmbeddings.shape.last, 500, 5)
    val contextVector = causalAttention.apply(batchEmbeddings)
    val multiHeadContextVector = multiheadAttention.apply(batchEmbeddings)
    println("Context Vectors")
    println(contextVector)
    println("Multi-headed Context Vectors")
    println(multiHeadContextVector)


