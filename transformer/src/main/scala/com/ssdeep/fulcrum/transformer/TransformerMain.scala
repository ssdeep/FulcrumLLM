package com.ssdeep.fulcrum.transformer

import com.ssdeep.fulcrum.tokenizer.Tokenizer
import com.ssdeep.fulcrum.transformer.MiniGPT
import com.ssdeep.fulcrum.transformer.config.GPTConfig
import com.ssdeep.fulcrum.transformer.nn.{FeedForward, NLayeredDeepNeuralNetwork}
import com.ssdeep.fulcrum.transformer.normalization.LayerNormalization
import com.typesafe.config.ConfigFactory
import torch.{Float32, Tensor}

object TransformerMain:
  def meanVar(x: Tensor[Float32]): (Tensor[Float32], Tensor[Float32]) =
    (
      torch.mean(x, dim = -1, keepdim = true, dtype = torch.float32),
      torch.variance(x, dim = -1, keepdim = true)
    )
  def main(args: Array[String]): Unit = {
    val config = ConfigFactory.load()
    val gptConfig = GPTConfig.loadFromConfig(config)
    val miniGpt = new MiniGPT(gptConfig)
    val tokenizer = Tokenizer.buildDefaultO200kBase
    val input = torch.Tensor.apply(tokenizer.encode("Some random text"))
    val input2 = torch.Tensor.apply(tokenizer.encode("Other random text"))
    val stackedInput = torch.stack(Seq(input, input2), dim = 0)
    val output = miniGpt.apply(stackedInput)
    torch.manualSeed(123L)
    val batchExample = torch.randn(Seq(2, 5), dtype = torch.float32)
    val layer = new torch.nn.Sequential(torch.nn.Linear(5, 6), torch.nn.ReLU())
    val out = layer(batchExample)
    val mean = out.mean(dim = -1, keepdim = true)
    val variance = torch.variance(out, dim = -1, keepdim = true)

    println(out)
    println(s"Mean ${mean}")
    println(s"Variance ${variance}")
    val outnorm = (out-mean)/torch.sqrt(variance)
    println(outnorm)
    println(s"Mean ${outnorm.mean(dim = -1)}")
    println(s"Variance ${torch.variance(outnorm, dim = -1)}")
    val layerNorm = new LayerNormalization(gptConfig.embedDim)
    val batchExample2 = torch.randn(Seq(2, gptConfig.embedDim))
    val answers = layerNorm(batchExample2)
    println("Layers normalized")
    println(answers)
    val (mean1, variance1) = meanVar(answers)
    println("Mean")
    println(mean1)
    println("Variance")
    println(variance1)
    val ffn = FeedForward(gptConfig)(answers)
    println("Feed Forward")
    println(ffn)
    val nLayeredModel = NLayeredDeepNeuralNetwork(Array(gptConfig.embedDim, gptConfig.embedDim, gptConfig.embedDim, 1), true)
    val nLayeredNoShortCut = NLayeredDeepNeuralNetwork(Array(gptConfig.embedDim, gptConfig.embedDim, gptConfig.embedDim, 1),
      false)
    val copyAnswers = layerNorm(batchExample2)
    println("Original answers")
    nLayeredModel.print_gradients(answers)
    println("copy answers")
    nLayeredNoShortCut.print_gradients(copyAnswers)

//    println(s"Variance: ${out}")
  }
