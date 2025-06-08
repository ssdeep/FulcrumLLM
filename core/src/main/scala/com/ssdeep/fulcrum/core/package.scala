package com.ssdeep.fulcrum

import me.tongfei.progressbar.{ProgressBar, ProgressBarBuilder}

import scala.jdk.CollectionConverters.*
package object core {
  
  def withProgressBuilder[T](name: String, iter: Iterator[T], length: Option[Int] = None): Iterator[T] =
    val pbBuilder = new ProgressBarBuilder
    pbBuilder.setTaskName(name)
    if (length.nonEmpty) {
      pbBuilder.setInitialMax(length.get)
    }
    ProgressBar.wrap(iter.asJava, pbBuilder).asScala
}
