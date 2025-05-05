import com.ssdeep.fulcrum.attention.SimpleAttention
import com.ssdeep.fulcrum.core.ClassNameLogger
import com.ssdeep.fulcrum.embedding.GPTDatasetV1
import com.ssdeep.fulcrum.tokenizer.Tokenizer
import torch.{Float32, Tensor}

import scala.io.Source

object Main extends ClassNameLogger:
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

    val simpleAttention = SimpleAttention(gptDataset)
    println(simpleAttention.getAttention)


