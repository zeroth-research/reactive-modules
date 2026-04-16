/-
Copyright (c) 2024. All rights reserved.
Released under Apache 2.0 license.
Authors: Your Name

Simple JSON parsing utilities.
-/
import Lean.Data.Json

namespace LeanAI.JSON

open Lean

/-- Extract a string field from a JSON object -/
def getString (json : Json) (key : String) : Option String :=
  match json.getObjVal? key with
  | Except.ok val =>
      match val.getStr? with
      | Except.ok str => some str
      | Except.error _ => none
  | Except.error _ => none

/-- Parse JSON string -/
def parse (s : String) : Except String Json :=
  Json.parse s

/-- Escape special characters for JSON strings -/
def escapeString (s : String) : String :=
  s.replace "\\" "\\\\"
    |>.replace "\"" "\\\""
    |>.replace "\n" "\\n"
    |>.replace "\r" "\\r"
    |>.replace "\t" "\\t"

/-- Build a simple JSON object string -/
def mkObject (fields : List (String × String)) : String :=
  let pairs := fields.map fun (k, v) => s!"\"{k}\": \"{escapeString v}\""
  "{" ++ String.intercalate ", " pairs ++ "}"

/-- Build a JSON object with proper type handling -/
def mkObjectRaw (fields : List (String × String)) : String :=
  let pairs := fields.map fun (k, v) =>
    -- Check if value is a boolean or number (doesn't need quotes)
    if v == "true" || v == "false" || v.all (fun c => c.isDigit || c == '.' || c == '-') then
      s!"\"{k}\": {v}"
    else
      s!"\"{k}\": \"{escapeString v}\""
  "{" ++ String.intercalate ", " pairs ++ "}"

end LeanAI.JSON
