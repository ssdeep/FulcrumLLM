val PekkoVersion = "1.1.3"
libraryDependencies ++= Seq(
  "org.apache.pekko" %% "pekko-stream" % PekkoVersion,
  "org.apache.pekko" %% "pekko-stream-testkit" % PekkoVersion % Test
)
libraryDependencies += "org.vegas-viz" %% "vegas" % "0.3.10"