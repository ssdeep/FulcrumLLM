import sbt._

object Dependencies {
  lazy val json4sNative = "org.json4s" %% "json4s-native" % "4.1.0-M8"
  lazy val log4jVersion = "2.24.3"
  lazy val log4jbom = "org.apache.logging.log4j" % "log4j-bom" % log4jVersion
  lazy val log4j = Seq(
    log4jbom,
    "org.apache.logging.log4j" % "log4j-api" % log4jVersion,
    "org.apache.logging.log4j" % "log4j-core" % log4jVersion,
    "org.apache.logging.log4j" % "log4j-layout-template-json" % log4jVersion
  )

}

