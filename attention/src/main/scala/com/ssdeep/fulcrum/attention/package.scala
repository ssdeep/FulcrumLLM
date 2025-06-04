package com.ssdeep.fulcrum
import org.bytedeco.pytorch.LongOptional
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

  /**
   * Extract a 2D tensor from a larger 2D tensor spanning from (0 to row, 0 to col)
   * Equivalent to python tensor[:row, :col]
   */
  extension[T <: DType](t: Tensor[T]) def span(row: Option[Long], col: Option[Long]): Tensor[T] = {
    require(t.shape.length == 2)
    val Seq(rowMax, colMax) = t.shape
    val rowSpan = row.map {
      case p if Math.abs(p) > rowMax => LongOptional()
      case p => LongOptional(p)
    }.getOrElse(LongOptional())
    val colSpan = col.map {
      case p if Math.abs(p) > colMax => LongOptional()
      case p => LongOptional(p)
    }.getOrElse(LongOptional())
    fromNative(
      t.native
      .slice(0L, LongOptional(), rowSpan, 1L)
      .slice(1L, LongOptional(), colSpan, 1L)
    )
  }
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
