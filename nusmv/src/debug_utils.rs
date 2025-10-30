use std::fmt::Debug;

/// Compares two Debug-printable values line by line, highlighting differences.
/// This is super helpful when comparing manually built modules vs parsed ones.
pub fn compare_debug<T: Debug, U: Debug>(label_a: &str, a: &T, label_b: &str, b: &U) {

    let a_str = format!("{:#?}", a);
    let b_str = format!("{:#?}", b);

    let a_lines: Vec<_> = a_str.lines().collect();
    let b_lines: Vec<_> = b_str.lines().collect();

    println!("==================== Comparing {} vs {} ====================", label_a, label_b);
    for (i, (l_a, l_b)) in a_lines.iter().zip(b_lines.iter()).enumerate() {
        if l_a.trim() != l_b.trim() {
            println!("❌ Line {} differs:", i + 1);
            println!("  {}: {}", label_a, l_a);
            println!("  {}: {}", label_b, l_b);
        }
    }

    // if one has more lines
    if a_lines.len() != b_lines.len() {
        println!("\n⚠️ Different number of lines ({} vs {})", a_lines.len(), b_lines.len());
    }
    println!("==============================================================");
}
