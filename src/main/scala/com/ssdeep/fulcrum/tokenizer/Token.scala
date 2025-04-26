package com.ssdeep.fulcrum.tokenizer

sealed trait Token:
  lazy val head: Token = this match
    case BytePair(first, second) => first.head
    case a @ UnitToken(c) => a
    case Empty => Empty
  lazy val tail: Token = this match
    case BytePair(first: UnitToken, second) => second
    case BytePair(first: BytePair, second) => BytePair(first.tail, second)
    case UnitToken(c) => Empty
    case Empty => Empty


  def isMatch(s: String): Boolean = this match
    case BytePair(first, second) => (0 until s.length).exists {
      i =>
        s.splitAt(i) match
          case (a, b) => first.isMatch(a) && second.isMatch(b)
    }
    case UnitToken(c) => c.toString == s
    case Empty => s == null || s.isEmpty


case class BytePair(first: Token, second: Token) extends Token

case class UnitToken(c: Char) extends Token
case object Empty extends Token

object Token:
   extension(t: Token) def stringify: String =
     t match
       case BytePair(first, second) => s"${first.stringify}${second.stringify}"
       case UnitToken(c) => s"$c"
       case Empty => ""
