ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "3.3.5"

lazy val root = (project in file("."))
  .settings(
    name := "FulcrumLLM"//,
//    idePackagePrefix := Some("com.ssdeep.fulcrum")
  )
