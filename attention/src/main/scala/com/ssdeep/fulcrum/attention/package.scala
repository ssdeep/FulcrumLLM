package com.ssdeep.fulcrum
import torch.*
import torch.nn.*
import org.bytedeco.pytorch.global.torch as pytorch
import torch.Tensor.fromNative

import java.awt.datatransfer.FlavorListener

package object attention {
  /**
   * 2D dot product, zip across the individual rows of the tensors and then dot product each row.
   * Single row of the Tensor produces a float value. All rows produce a 1D Tensor of float values.
   */
  extension(t: Tensor[Float32]) def dot(j: Tensor[Float32]): Tensor[Float32] =
    val newVector = (0 until t.shape(0)).zip(0 until j.shape(0)).map {
      case (ti, ji) => pytorch.dot(t(ti).native, j(ji).native).item_float()
    }
    torch.Tensor.apply(newVector)

  extension(t: Tensor[Float32]) def dot2d(j: Tensor[Float32]): Tensor[Float32] =
    val newVector = (0 until t.shape(0)).zip(0 until j.shape(0)).map {
      case (ti, ji) =>
        t(ti).dot(j(ji)).toSeq
    }
    torch.Tensor.apply(newVector)

  extension (t: Tensor[Float32]) def %(o: Tensor[Float32]): Tensor[Float32] =
    fromNative(t.native.matmul(o.transpose(0,1).native))

  extension [T <: FloatNN](t: Tensor[T]) def map(f: Tensor[T] => Tensor[T]): Tensor[T] =
    if t.shape.length > 1 then
       val rows = (0 until t.shape.head).map {
         i =>
           val tensor = t(i)
           tensor.map(f).toSeq.map(_.asInstanceOf[Float])
       }
       torch.Tensor.apply(rows).asInstanceOf[Tensor[T]]
    else f(t)



}
