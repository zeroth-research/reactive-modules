use std::fs::File;
use std::io::Write;

use base::{Interface, Term};

use common::transition::{Transition, WiredTransitions};

const CSS: &str = r#"
body {
    font-family: system-ui, sans-serif;
    background: #f5f5f5;
}

#container {
    max-width: 800px;
    margin: 40px auto;
    display: flex;
    flex-direction: column;
    gap: 16px;
}

details {
    background: white;
    border-radius: 8px;
    padding-left: 16px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.1);
}

summary {
    cursor: pointer;
    font-weight: 600;
    list-style: none;
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px;
   border-bottom: 1px solid #ddd;
}

summary::-webkit-details-marker {
    display: none;
}

/* Arrow indicator */
summary::before {
    content: "";
    width: 0;
    height: 0;
    border-left: 6px solid #333;
    border-top: 5px solid transparent;
    border-bottom: 5px solid transparent;
    transition: transform 200ms ease;
}

details[open] summary::before {
    transform: rotate(90deg);
}

/* Dynamic-height transition using CSS Grid */
.details-grid {
    display: grid;
    grid-template-rows: 0fr;
    transition: grid-template-rows 300ms ease, opacity 200ms ease;
    opacity: 0;
}

details[open] .details-grid {
    grid-template-rows: 1fr;
    opacity: 1;
    margin-top: 12px;
}

.details-content {
   //overflow: hidden;
   //padding-top: 12px;
   // border-top: 1px solid #ddd;
   //white-space: pre-wrap;
   margin: 0px;
}
"#;

fn format_term<D, I>(term: &base::Term<D, I>) -> String
where
    D: std::fmt::Display,
    I: std::fmt::Display,
{
    let color_term_lhs = "<span style=\"color: #106EE2\">";
    let color_term_op = "<span style=\"color: #E88914\">";
    let color_clr = "</span>";

    let reads = term
        .read()
        .ids()
        .map(|id| format!("x{id}"))
        .collect::<Vec<String>>()
        .join(", ");

    let writes = term
        .write()
        .ids()
        .map(|id| format!("x{id}"))
        .collect::<Vec<String>>()
        .join(", ");

    format!(
        "{color_term_lhs}{writes}{color_clr} = {color_term_op}{}{color_clr}({reads})",
        term.itype()
    )
}

fn format_terms<D, I>(tr: &Transition<D, I>) -> String
where
    D: std::fmt::Display,
    I: std::fmt::Display,
{
    let terms = tr
        .terms()
        .map(|t| format_term(t))
        .collect::<Vec<String>>()
        .join("\n");

    format!("<pre>{terms}</pre>")
}

fn format_intf<D: std::fmt::Display, const N: usize>(intf: &Interface<D, N>) -> String {
    intf.iter()
        .map(|elem| {
            let s = elem
                .iter()
                //.map(|w| format!("<span style=\"color: #106EE2\">x{}</span>", w.id()))
                .map(|w| format!("<span style=\"color: black\">x{}</span>", w.id()))
                .collect::<Vec<String>>()
                .join(", ");
            format!(
                "{}: <span style=\"color: #f54242\">{}</span>",
                s,
                elem[0].dtype()
            )
        })
        .collect::<Vec<String>>()
        .join(", ")
}

pub fn write_to_html<D, I>(
    transitions: &WiredTransitions<D, I>,
    path: &str,
) -> Result<(), std::io::Error>
where
    //Term<D, I>: std::fmt::Display,
    I: std::fmt::Display,
    D: std::fmt::Display,
{
    let items: Vec<String> = transitions
        .iter()
        .map(|tr| {
            let intf_in = format_intf(tr.intf_in());
            let intf_env = format_intf(tr.intf_env().unwrap_or(&base::Interface::empty()));
            let intf_out = format_intf(tr.intf_out());

            format!(
                r#"
<details>
  <summary>
  <table>
  <tr><td style="color:blue"> In: </td><td>{}</td></tr>
  <tr><td style="color:blue"> Env:</td><td>{}</td></tr>
  <tr style="border-top: 1pt solid black">
      <td style="color:blue"> Out:</td><td>{}</td></tr>
  </table>
  </summary>

  <div class="details-grid">
    <div class="details-content">
    <pre>In : {}
Env: {}</pre>
    {}
    <pre>Out: {}</pre>
    </div>
  </div>
</details>
"#,
                &intf_in,
                &intf_env,
                &intf_out,
                &intf_in,
                &intf_env,
                format_terms(tr),
                &intf_out
            )
        })
        .collect();

    let html = format!(
        r#"
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Unrolling</title>

<style>
{CSS}
</style>

</head>
<body>
<div id="container">
{}
</div> <!-- container end -->
</body>
</html>
"#,
        items.join("\n\n")
    );

    File::create(path)?.write_all(html.as_bytes())?;
    Ok(())
}
