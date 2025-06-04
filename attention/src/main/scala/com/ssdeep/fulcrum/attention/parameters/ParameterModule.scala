package com.ssdeep.fulcrum.attention.parameters

import torch._
import torch.nn._
import org.bytedeco.pytorch.global.torch as pytorch

class ParameterModule(dimensionX: Int, dimensionY: Int) extends Module:
  val weightQ = torch.randn(Seq(dimensionX, dimensionY))
  val weightK = torch.randn(Seq(dimensionX, dimensionY))
  val weightV = torch.randn(Seq(dimensionX, dimensionY))
  def processInput(input: Tensor[Float32]) =
    val Wq = input `@` weightQ
    val Wk = input `@` weightK
    val Wv = input `@` weightV
    println("Weight  Q")
    println(Wq)
    println("Weight K")
    println(Wk)
    println("Weight V")
    println(Wv)
    Wq


