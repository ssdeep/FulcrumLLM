ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "3.3.5"

val json4sNative = "org.json4s" %% "json4s-native" % "4.1.0-M8"

libraryDependencies += json4sNative
libraryDependencies += "me.tongfei" % "progressbar" % "0.10.1"

lazy val root = (project in file("."))
  .settings(
    name := "FulcrumLLM"//,
//    idePackagePrefix := Some("com.ssdeep.fulcrum")
  )
