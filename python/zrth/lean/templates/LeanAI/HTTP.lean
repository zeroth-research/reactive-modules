/-
Copyright (c) 2024. All rights reserved.
Released under Apache 2.0 license.
Authors: Your Name

Simple HTTP client using curl subprocess.
-/

namespace LeanAI.HTTP

/-- Make an HTTP POST request using curl with custom headers -/
def postWithHeaders (url : String) (body : String) (headers : List (String × String)) : IO String := do
  -- Build curl command with headers
  let mut args := #[
    "-s",           -- Silent mode
    "-X", "POST",   -- POST request
    "-H", "Content-Type: application/json",  -- JSON content type
    "-d", body  -- Request body
  ]

  -- Add custom headers
  for (key, value) in headers do
    args := args.push "-H"
    args := args.push s!"{key}: {value}"

  -- Add URL last
  args := args.push url

  -- Execute curl
  let output ← IO.Process.output {
    cmd := "curl"
    args := args
  }

  if output.exitCode != 0 then
    throw (IO.userError s!"HTTP request failed: {output.stderr}")

  return output.stdout.trim

/-- Make an HTTP POST request using curl (convenience wrapper without headers) -/
def post (url : String) (body : String) : IO String :=
  postWithHeaders url body []

/-- Make an HTTP GET request using curl -/
def get (url : String) (headers : List (String × String) := []) : IO String := do
  let mut args := #["-s"]  -- Silent mode

  -- Add custom headers
  for (key, value) in headers do
    args := args.push "-H"
    args := args.push s!"{key}: {value}"

  args := args.push url

  let output ← IO.Process.output {
    cmd := "curl"
    args := args
  }

  if output.exitCode != 0 then
    throw (IO.userError s!"HTTP request failed: {output.stderr}")

  return output.stdout.trim

end LeanAI.HTTP
