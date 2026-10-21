#!/usr/bin/env Rscript
# s04_run_chromvar.R -- per-species chromVAR on the all-QC-peak pseudobulk matrix.
#
# WHY this step exists (readme sections 17/18): chromVAR is the independent
# raw-accessibility validation of the model-derived motif programs.  chromVAR
# needs two things the pseudobulk export (s03) cannot provide -- a per-peak GC
# bias and a binary motif x peak annotation -- and both are derived here
# directly from the species FASTA, because no BSgenome exists for calJac4 /
# rheMac8 / monDom5 / panPan1.1 and section 18 explicitly forbids building one.
# The GC fraction is computed exactly as the Python side defines it (G+C over
# peak length, 0-based half-open coordinates), so the two halves of the pipeline
# see the same sequences.
#
# Reads   results/03_chromvar_input/<species>/{counts.mtx.gz,peaks.tsv.gz,
#         pseudobulk_metadata.tsv}
# Writes  results/04_chromvar/<species>/{chromvar_z.tsv.gz,motif_peak_matches.rds,
#         gc_bias.tsv.gz,chromvar_qc.txt,chromvar_deviation_scores.rds}
#         results/04_chromvar/chromvar_run_audit.tsv   (one row per species)
#
# Usage:
#   Rscript script/s04_run_chromvar.R --config config/analysis.yaml \
#       [--species human,mouse] [--input-root DIR] [--out-root DIR] \
#       [--genome-table FILE]
#
# --input-root / --out-root / --genome-table default to the frozen contract
# paths and exist only so that a smoke test can run against a downsampled tree
# outside results/.  With no extra flags the script is exactly the contract.

suppressPackageStartupMessages({
  library(Biostrings)
  library(Matrix)
  library(S4Vectors)
  library(SummarizedExperiment)
  library(TFBSTools)
  library(chromVAR)
  library(motifmatchr)
})

PYTHON_BIN <- "python"
DEFAULT_CONFIG <- "config/analysis.yaml"
CHROMVAR_STEP <- "04_chromvar"
INPUT_STEP <- "03_chromvar_input"


log_line <- function(...) {
  cat(sprintf("[%s]", format(Sys.time(), "%H:%M:%S")), ..., "\n", sep = " ", file = stderr())
}


# ---------------------------------------------------------------- arguments


parse_args <- function(argv) {
  out <- list(config = NULL, species = NULL, input_root = NULL, out_root = NULL, genome_table = NULL)
  i <- 1L
  while (i <= length(argv)) {
    token <- argv[[i]]
    if (grepl("^--[a-z-]+=", token)) {
      key <- sub("^--", "", sub("=.*$", "", token))
      value <- sub("^[^=]*=", "", token)
    } else if (grepl("^--", token)) {
      key <- sub("^--", "", token)
      i <- i + 1L
      if (i > length(argv)) stop("missing value for --", key, call. = FALSE)
      value <- argv[[i]]
    } else {
      stop("unexpected positional argument: ", token, call. = FALSE)
    }
    key <- gsub("-", "_", key)
    if (!key %in% names(out)) stop("unknown argument --", key, call. = FALSE)
    out[[key]] <- value
    i <- i + 1L
  }
  out
}


project_root <- function() {
  file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(file_arg) == 0L) stop("cannot locate this script; run it with Rscript", call. = FALSE)
  script_path <- normalizePath(sub("^--file=", "", file_arg[[1]]))
  dirname(dirname(script_path))
}


# The analysis config is a nested YAML map.  Rather than depend on a YAML or
# JSON package, ask the frozen python environment for the handful of scalars
# this step needs and read them back as a two-column TSV.
config_scalars <- function(config_path) {
  code <- paste(
    "import sys, yaml",
    "c = yaml.safe_load(open(sys.argv[1]))",
    "rows = [",
    "  ('paths.results', c['paths']['results']),",
    "  ('motif_database.file', c['motif_database']['file']),",
    "  ('motif_database.n_motifs', c['motif_database']['n_motifs']),",
    "  ('chromvar.seed', c['chromvar']['seed']),",
    "  ('chromvar.min_cells_per_pseudobulk', c['chromvar']['min_cells_per_pseudobulk']),",
    "]",
    "sys.stdout.write(''.join('%s\\t%s\\n' % (k, v) for k, v in rows))",
    sep = "\n"
  )
  raw <- suppressWarnings(
    system2(PYTHON_BIN, c("-c", shQuote(code), shQuote(config_path)),
            stdout = TRUE, stderr = FALSE)
  )
  if (length(raw) == 0L || !any(grepl("\t", raw))) {
    stop("failed to read scalars from ", config_path, " through ", PYTHON_BIN, call. = FALSE)
  }
  fields <- strsplit(raw[grepl("\t", raw)], "\t", fixed = TRUE)
  values <- setNames(lapply(fields, function(f) paste(f[-1], collapse = "\t")),
                     vapply(fields, `[[`, character(1), 1L))
  values
}


read_species_table <- function(table_path, root) {
  table <- read.delim(table_path, stringsAsFactors = FALSE, check.names = FALSE)
  required <- c("species", "assembly", "fasta", "prefix", "chrom_style")
  missing <- setdiff(required, names(table))
  if (length(missing)) {
    stop("species_genome.tsv lacks columns: ", paste(missing, collapse = ", "), call. = FALSE)
  }
  table$fasta_path <- ifelse(grepl("^/", table$fasta), table$fasta, file.path(root, table$fasta))
  table
}


count_bases <- function(sequence, base) {
  nchar(sequence) - nchar(gsub(base, "", sequence, fixed = TRUE))
}


# ------------------------------------------------------- peak -> reference

# Mirrors common.parse_peak: strip the assembly prefix, then add the `chr`
# prefix when the assembly stores contigs without it (macaque / rheMac8).
# Unparseable ids are returned with chrom = NA rather than dropped, so the
# caller can report them.
parse_peaks <- function(peak_ids, prefix, add_chr) {
  body <- ifelse(startsWith(peak_ids, prefix), substring(peak_ids, nchar(prefix) + 1L), peak_ids)
  parts <- regmatches(body, regexec("^([^:]+):([0-9]+)-([0-9]+)$", body))
  usable <- lengths(parts) == 4L
  frame <- data.frame(peak = peak_ids, chrom = NA_character_, start = NA_real_, end = NA_real_,
                      stringsAsFactors = FALSE)
  if (any(usable)) {
    chrom <- vapply(parts[usable], `[[`, character(1), 2L)
    if (add_chr) chrom <- ifelse(startsWith(chrom, "chr"), chrom, paste0("chr", chrom))
    frame$chrom[usable] <- chrom
    frame$start[usable] <- as.numeric(vapply(parts[usable], `[[`, character(1), 3L))
    frame$end[usable] <- as.numeric(vapply(parts[usable], `[[`, character(1), 4L))
  }
  frame
}


# Stream the gzipped FASTA once and slice out only the peaks we asked for.
# Chromosome strings are released as soon as their peaks are extracted, so peak
# memory stays at (largest contig + total peak sequence), not at genome size.
# Coordinates follow common.read_sequences: 0-based half-open start, so the R
# substring is [start + 1, end].
extract_sequences <- function(fasta_path, coords, chunk_lines = 200000L) {
  if (nrow(coords) == 0L) {
    return(list(sequences = character(0), chroms_found = character(0), chroms_wanted = character(0)))
  }
  wanted <- unique(coords$chrom)
  by_chrom <- split(coords, coords$chrom)
  sequences <- setNames(rep(NA_character_, nrow(coords)), coords$peak)
  seen <- character(0)

  current <- NULL
  keep_current <- FALSE
  # Each element is one already-collapsed run of sequence lines. Collapsing per
  # chunk keeps the accumulator to a handful of large strings instead of the
  # millions of small ones readLines hands back.
  pieces <- list()
  n_pieces <- 0L
  push <- function(lines) {
    if (length(lines) == 0L) return(invisible(NULL))
    n_pieces <<- n_pieces + 1L
    pieces[[n_pieces]] <<- paste(lines, collapse = "")
    invisible(NULL)
  }
  flush <- function() {
    if (is.null(current) || !keep_current || n_pieces == 0L) return(invisible(NULL))
    sequence <- paste(unlist(pieces, use.names = FALSE), collapse = "")
    entries <- by_chrom[[current]]
    start <- entries$start
    end <- entries$end
    usable <- start >= 0 & start < end & end <= nchar(sequence)
    if (any(usable)) {
      # substring(), not substr(): substr() ignores all but the first
      # start/stop pair when the subject has length one.
      sequences[entries$peak[usable]] <<- toupper(substring(sequence, start[usable] + 1, end[usable]))
    }
    pieces <<- list()
    n_pieces <<- 0L
    invisible(NULL)
  }

  connection <- gzfile(fasta_path, "rt")
  on.exit(close(connection), add = TRUE)
  repeat {
    lines <- readLines(connection, n = chunk_lines, warn = FALSE)
    if (length(lines) == 0L) {
      flush()
      break
    }
    headers <- which(startsWith(lines, ">"))
    if (length(headers) == 0L) {
      if (keep_current) push(lines)
    } else {
      if (headers[[1]] > 1L && keep_current) push(lines[seq_len(headers[[1]] - 1L)])
      for (index in seq_along(headers)) {
        flush()
        current <- strsplit(sub("^>", "", lines[[headers[[index]]]]), "[ \t]+")[[1]][[1]]
        keep_current <- current %in% wanted
        if (keep_current) seen <- c(seen, current)
        to <- if (index < length(headers)) headers[[index + 1L]] - 1L else length(lines)
        from <- headers[[index]] + 1L
        if (to >= from && keep_current) push(lines[from:to])
      }
    }
    # Safe to stop only once a contig we do not need has been reached: that
    # proves every wanted contig has been read to its end.
    if (!keep_current && all(wanted %in% seen)) {
      flush()
      break
    }
  }
  list(sequences = sequences, chroms_found = unique(seen), chroms_wanted = wanted)
}


# ------------------------------------------------------------------- motifs

# The JASPAR2024 file is MEME text, which TFBSTools::readJASPARMatrix cannot
# parse (it only accepts the 5-line "all" JASPAR format), so the PFMs are read
# here with the same field layout the python side's motif_catalog uses.
#
# The MEME body holds per-position PROBABILITIES, and TFBSTools::PFMatrix()
# starts with `mode(profileMatrix) <- "integer"`: handing it probabilities
# truncates every value below 1.0 to zero, leaves all-zero PWM columns and
# produces a motif annotation that has nothing to do with JASPAR.  The values
# are therefore rescaled to counts with the record's own nsites first, which
# gives the same PFM readJASPARMatrix() builds from a JASPAR "all" file; toPWM()
# then applies the usual 0.8 pseudocount against the uniform 0.25 background.
read_meme_pfms <- function(path) {
  lines <- readLines(path)
  starts <- grep("^MOTIF[ \t]", lines)
  if (length(starts) == 0L) stop("no MOTIF records in ", path, call. = FALSE)
  ids <- sub("^MOTIF[ \t]+([^ \t]+).*$", "\\1", lines[starts])
  labels <- trimws(sub("^MOTIF[ \t]+[^ \t]+", "", lines[starts]))
  numeric_row <- "^[ \t]*[-0-9.eE]+([ \t]+[-0-9.eE]+){3}[ \t]*$"
  matrices <- vector("list", length(starts))
  for (index in seq_along(starts)) {
    stop_at <- if (index < length(starts)) starts[[index + 1L]] - 1L else length(lines)
    block <- lines[(starts[[index]] + 1L):stop_at]
    declared <- grep("letter-probability matrix", block, value = TRUE)
    if (length(declared) == 0L) {
      stop("motif ", ids[[index]], " has no letter-probability matrix header", call. = FALSE)
    }
    nsites <- suppressWarnings(as.numeric(sub("^.*nsites=[ \t]*([0-9]+).*$", "\\1", declared[[1]])))
    if (!is.finite(nsites) || nsites < 1) {
      stop("motif ", ids[[index]], " has no usable nsites", call. = FALSE)
    }
    body <- block[grepl(numeric_row, block)]
    if (length(body) == 0L) stop("motif ", ids[[index]], " has no probability row", call. = FALSE)
    values <- as.numeric(unlist(strsplit(trimws(body), "[ \t]+")))
    width <- length(values) / 4
    profile <- matrix(values, nrow = 4L, ncol = width)
    rownames(profile) <- DNA_BASES
    # The body must be a probability matrix: if it is not, this reader was fed
    # the wrong file and every count below would be an artifact.
    if (max(abs(colSums(profile) - 1)) > 1e-3) {
      stop("motif ", ids[[index]], " does not hold a letter-probability matrix", call. = FALSE)
    }
    motif <- PFMatrix(ID = ids[[index]],
                      name = if (nzchar(labels[[index]])) labels[[index]] else ids[[index]],
                      matrixClass = "PFM",
                      profileMatrix = round(profile * nsites))
    # Read back what PFMatrix() actually stored: with probabilities it would be
    # truncated to at most one count per column, so this fires on the regression.
    if (max(colSums(as.matrix(motif@profileMatrix))) <= 1) {
      stop("motif ", ids[[index]], " collapsed to at most one count per column; ",
           "PFMatrix() cannot hold probabilities", call. = FALSE)
    }
    matrices[[index]] <- motif
  }
  names(matrices) <- ids
  do.call(PFMatrixList, matrices)
}


# Guards the regression above on the objects that are actually used downstream:
# a PFMatrix built from probabilities has at most one count per column, and the
# log-odds of such a matrix collapses to a couple of distinct values.  Both are
# true for the real JASPAR counts (879 PFMs, median 32 distinct PWM values) and
# both fail on truncated probabilities.  Returns the QC values written to
# chromvar_qc.txt.
check_motif_matrices <- function(pfms, pwms, species) {
  pfm_column_max <- vapply(pfms, function(p) max(colSums(as.matrix(p@profileMatrix))), numeric(1))
  if (any(pfm_column_max <= 1)) {
    stop(sprintf("%s: %d PFM(s) hold at most one count per column; PFMatrix() was given probabilities",
                 species, sum(pfm_column_max <= 1)), call. = FALSE)
  }
  pwm_distinct <- vapply(pwms, function(p) length(unique(round(as.vector(p@profileMatrix), 6))), numeric(1))
  if (stats::median(pwm_distinct) <= 3) {
    stop(sprintf("%s: only %.0f distinct PWM values per motif; the log-odds conversion is degenerate",
                 species, stats::median(pwm_distinct)), call. = FALSE)
  }
  list(pfm_min_column_max = min(pfm_column_max),
       pfm_median_column_max = stats::median(pfm_column_max),
       pwm_min_distinct = min(pwm_distinct),
       pwm_median_distinct = stats::median(pwm_distinct))
}


# -------------------------------------------------------------------- input

read_mtx <- function(path, n_peaks, n_pseudobulk) {
  matrix <- as(readMM(gzfile(path)), "CsparseMatrix")
  if (nrow(matrix) == n_peaks && ncol(matrix) == n_pseudobulk) return(list(matrix = matrix, transposed = FALSE))
  if (nrow(matrix) == n_pseudobulk && ncol(matrix) == n_peaks) {
    return(list(matrix = t(matrix), transposed = TRUE))
  }
  stop(sprintf("counts.mtx.gz is %d x %d, expected %d peaks x %d pseudobulks",
               nrow(matrix), ncol(matrix), n_peaks, n_pseudobulk), call. = FALSE)
}


read_gz_table <- function(path, header = TRUE) {
  read.delim(gzfile(path), header = header, stringsAsFactors = FALSE,
             check.names = FALSE, colClasses = "character")
}


# ------------------------------------------------------------------- output

write_matrix_gz <- function(matrix, path) {
  connection <- gzfile(path, "wt")
  on.exit(close(connection), add = TRUE)
  write.table(matrix, connection, sep = "\t", quote = FALSE, col.names = NA)
}


write_table_gz <- function(frame, path) {
  connection <- gzfile(path, "wt")
  on.exit(close(connection), add = TRUE)
  write.table(frame, connection, sep = "\t", quote = FALSE, row.names = FALSE)
}


# ---------------------------------------------------------------- one species

# The audit row is the only place a lost species or a dropped peak can show up,
# so every branch of run_species has to return the same columns -- including the
# one that reports a species that died on an unexpected error.
blank_audit_row <- function(species, status, note) {
  data.frame(species = species, status = status, n_peaks = NA_integer_,
             n_peaks_with_sequence = NA_integer_, n_peaks_dropped_no_sequence = NA_integer_,
             n_peaks_dropped_zero_counts = NA_integer_, n_pseudobulk = NA_integer_,
             n_motifs = NA_integer_, n_motifs_matched = NA_integer_,
             n_peaks_with_motif = NA_integer_,
             fraction_peaks_with_motif = NA_real_,
             n_na_z = NA_integer_, fraction_na_z = NA_real_,
             median_pseudobulk_counts = NA_real_,
             transposed_counts = NA, seconds = NA_real_, note = note,
             stringsAsFactors = FALSE)
}


run_species <- function(species, genome_row, scalars, input_root, out_root, root) {
  started <- Sys.time()
  input_dir <- file.path(input_root, species)
  needed <- c("counts.mtx.gz", "peaks.tsv.gz", "pseudobulk_metadata.tsv")
  present <- file.exists(file.path(input_dir, needed))
  if (!all(present)) {
    log_line("WARNING", species, "missing input(s):",
             paste(needed[!present], collapse = ", "))
    return(blank_audit_row(species, "skipped_missing_input",
                           paste("missing:", paste(needed[!present], collapse = ","))))
  }

  peaks <- read_gz_table(file.path(input_dir, "peaks.tsv.gz"))
  if (!identical(names(peaks)[[1]], "peak")) {
    stop(species, ": peaks.tsv.gz must carry the header 'peak'", call. = FALSE)
  }
  peak_ids <- peaks[[1]]
  metadata <- read.delim(file.path(input_dir, "pseudobulk_metadata.tsv"),
                         stringsAsFactors = FALSE, check.names = FALSE)
  for (column in c("pseudobulk", "stage", "celltype")) {
    if (!column %in% names(metadata)) {
      stop(species, ": pseudobulk_metadata.tsv lacks column ", column, call. = FALSE)
    }
  }
  pseudobulk_ids <- metadata$pseudobulk
  if (anyDuplicated(peak_ids) || anyDuplicated(pseudobulk_ids)) {
    stop(species, ": duplicated peak or pseudobulk ids in the s03 export", call. = FALSE)
  }
  log_line(species, "input", length(peak_ids), "peaks x", length(pseudobulk_ids), "pseudobulks")

  loaded <- read_mtx(file.path(input_dir, "counts.mtx.gz"), length(peak_ids), length(pseudobulk_ids))
  counts <- loaded$matrix
  rownames(counts) <- peak_ids
  colnames(counts) <- pseudobulk_ids

  # --- reference sequence + GC bias ---------------------------------------
  coords <- parse_peaks(peak_ids, genome_row$prefix, genome_row$chrom_style == "add_chr")
  unparseable <- is.na(coords$chrom)
  if (any(unparseable)) {
    log_line("WARNING", species, sum(unparseable), "peak ids could not be parsed; dropped")
  }
  extraction <- extract_sequences(genome_row$fasta_path, coords[!unparseable, , drop = FALSE])
  sequences <- extraction$sequences
  sequence <- ifelse(is.na(sequences[peak_ids]), "", sequences[peak_ids])
  names(sequence) <- peak_ids
  length_peak <- nchar(sequence)
  gc <- ifelse(length_peak > 0L, count_bases(sequence, "G") + count_bases(sequence, "C"), NA_real_) / length_peak
  with_sequence <- length_peak > 0L
  dropped_no_sequence <- peak_ids[!with_sequence]
  missing_chroms <- setdiff(extraction$chroms_wanted, extraction$chroms_found)

  warnings <- character(0)
  if (length(dropped_no_sequence)) {
    warnings <- c(warnings, sprintf("%d peak(s) had no extractable sequence and were removed from the chromVAR matrix: %s%s",
                                    length(dropped_no_sequence),
                                    paste(head(dropped_no_sequence, 5), collapse = ","),
                                    if (length(dropped_no_sequence) > 5) ",..." else ""))
  }
  if (length(missing_chroms)) {
    warnings <- c(warnings, sprintf("contig(s) absent from the FASTA: %s", paste(missing_chroms, collapse = ",")))
  }
  if (loaded$transposed) {
    warnings <- c(warnings, "counts.mtx.gz was pseudobulk x peak and was transposed")
  }
  # A peak with zero counts in every pseudobulk makes chromVAR stop("All peaks
  # must have at least one fragment in one sample"), which would kill the step
  # before its audit is written; such peaks are removed here, on the same terms
  # as peaks without sequence, and never silently.
  keep_with_sequence <- peak_ids[with_sequence]
  counts <- counts[keep_with_sequence, , drop = FALSE]
  has_counts <- Matrix::rowSums(counts) > 0
  dropped_zero_counts <- keep_with_sequence[!has_counts]
  if (length(dropped_zero_counts)) {
    warnings <- c(warnings, sprintf("%d peak(s) had zero counts in every pseudobulk and were removed from the chromVAR matrix: %s%s",
                                    length(dropped_zero_counts),
                                    paste(head(dropped_zero_counts, 5), collapse = ","),
                                    if (length(dropped_zero_counts) > 5) ",..." else ""))
  }
  keep <- keep_with_sequence[has_counts]
  if (length(keep) == 0L) {
    stop(species, ": no peak is left with both a reference sequence and a non-zero count", call. = FALSE)
  }
  counts <- counts[keep, , drop = FALSE]
  gc <- gc[match(keep, peak_ids)]
  sequence <- sequence[match(keep, peak_ids)]

  # --- motif annotation ----------------------------------------------------
  pfms <- read_meme_pfms(file.path(root, scalars[["motif_database.file"]]))
  expected_motifs <- as.integer(scalars[["motif_database.n_motifs"]])
  if (length(pfms) != expected_motifs) {
    stop(sprintf("%s holds %d motifs, config freezes %d",
                 scalars[["motif_database.file"]], length(pfms), expected_motifs), call. = FALSE)
  }
  # The log-odds PWM is the standard TFBSTools conversion of those counts: the
  # 0.8 pseudocount against the uniform 0.25 background.  TFBSTools 1.40.0 names
  # that type "log2probratio" (match.arg rejects "log2proportional") and requires
  # the background to be named; this is spelled out here rather than left to the
  # defaults, which resolve to the same values.
  pwms <- toPWM(pfms, type = "log2probratio", bg = c(A = 0.25, C = 0.25, G = 0.25, T = 0.25))
  names(pwms) <- names(pfms)
  motif_qc <- check_motif_matrices(pfms, pwms, species)

  subject <- DNAStringSet(sequence)
  names(subject) <- keep
  matches <- matchMotifs(pwms, subject, out = "matches")
  matched_ids <- colnames(assay(matches))
  match_matrix <- assay(matches)
  if (!identical(dim(match_matrix), c(length(keep), length(pwms)))) {
    stop(sprintf("%s: matchMotifs returned %d x %d, expected %d peaks x %d motifs",
                 species, nrow(match_matrix), ncol(match_matrix),
                 length(keep), length(pwms)), call. = FALSE)
  }
  # matchMotifs(out="matches") is peaks x motifs, so the two margins answer two
  # different questions and must never be reported under one name: the rows count
  # peaks carrying at least one motif, the columns count catalog motifs that are
  # used by at least one peak.
  peaks_with_motif <- Matrix::rowSums(match_matrix) > 0
  motifs_with_match <- Matrix::colSums(match_matrix) > 0

  # --- chromVAR ------------------------------------------------------------
  annotations <- SummarizedExperiment(assays = list(counts = counts))
  rowData(annotations)$bias <- gc
  colData(annotations)$pseudobulk <- pseudobulk_ids

  seed <- as.integer(scalars[["chromvar.seed"]])
  set.seed(seed)
  background <- getBackgroundPeaks(object = annotations)
  deviations <- computeDeviations(object = annotations, annotations = matches,
                                  background_peaks = background)

  z <- as.matrix(assay(deviations, "z"))
  if (!identical(rownames(z), matched_ids)) {
    stop(species, ": motif order of the deviation matrix does not follow the PWM list", call. = FALSE)
  }
  colnames(z) <- pseudobulk_ids

  # chromVAR sets z to NA wherever a motif's expected count in a pseudobulk falls
  # below its internal threshold, so the contracted z matrix is NA by design and
  # not necessarily by bug.  The amount is recorded here (qc + audit) so a
  # downstream step that meets NaN can attribute it to that filter.
  n_na_z <- sum(is.na(z))
  fraction_na_z <- n_na_z / length(z)
  na_pseudobulk_rows <- Matrix::colSums(is.na(z)) > 0
  if (n_na_z > 0L) {
    warnings <- c(warnings, sprintf(
      "%d of %d chromVAR z values are NA (expected motif count below chromVAR's threshold); %d/%d pseudobulk(s) affected, %d of them entirely: %s%s",
      n_na_z, length(z), sum(na_pseudobulk_rows), length(pseudobulk_ids),
      sum(Matrix::colSums(is.na(z)) == nrow(z)),
      paste(head(pseudobulk_ids[na_pseudobulk_rows], 5), collapse = ","),
      if (sum(na_pseudobulk_rows) > 5) ",..." else ""))
  }

  pseudobulk_totals <- Matrix::colSums(counts)
  median_total <- stats::median(pseudobulk_totals)
  if (any(pseudobulk_totals == 0)) {
    warnings <- c(warnings, sprintf("pseudobulk(s) with zero total counts: %s",
                                    paste(pseudobulk_ids[pseudobulk_totals == 0], collapse = ",")))
  }

  species_dir <- file.path(out_root, species)
  dir.create(species_dir, showWarnings = FALSE, recursive = TRUE)
  write_matrix_gz(z, file.path(species_dir, "chromvar_z.tsv.gz"))
  write_table_gz(data.frame(peak = keep, gc = gc, length = as.integer(length_peak[match(keep, peak_ids)])),
                 file.path(species_dir, "gc_bias.tsv.gz"))
  # matchMotifs(out="matches") leaves rownames(matches) NULL, so without this the
  # saved annotation would identify its peaks by position alone (joinable only
  # through gc_bias.tsv.gz's row order); the peak ids and their gc/length are
  # attached as rowData so the matrix can be joined by id.
  rownames(matches) <- keep
  rowData(matches) <- DataFrame(peak = keep, gc = as.numeric(gc),
                                length = as.integer(length_peak[match(keep, peak_ids)]))
  saveRDS(matches, file.path(species_dir, "motif_peak_matches.rds"))
  saveRDS(deviations, file.path(species_dir, "chromvar_deviation_scores.rds"))

  fraction_with_motif <- mean(peaks_with_motif)
  qc <- c(
    sprintf("species\t%s", species),
    sprintf("assembly\t%s", genome_row$assembly),
    sprintf("fasta\t%s", genome_row$fasta_path),
    sprintf("n_peaks\t%d", length(peak_ids)),
    sprintf("n_peaks_with_sequence\t%d", length(keep_with_sequence)),
    sprintf("n_peaks_dropped_no_sequence\t%d", length(dropped_no_sequence)),
    sprintf("n_peaks_dropped_zero_counts\t%d", length(dropped_zero_counts)),
    sprintf("n_peaks_in_chromvar_matrix\t%d", length(keep)),
    sprintf("n_pseudobulk\t%d", length(pseudobulk_ids)),
    sprintf("n_motifs_in_catalog\t%d", length(pfms)),
    sprintf("n_motifs_matched\t%d", sum(motifs_with_match)),
    sprintf("n_peaks_with_motif\t%d", sum(peaks_with_motif)),
    sprintf("fraction_peaks_matching_at_least_one_motif\t%.6f", fraction_with_motif),
    sprintf("n_na_z\t%d", n_na_z),
    sprintf("fraction_na_z\t%.6f", fraction_na_z),
    sprintf("median_pseudobulk_fragment_counts\t%.0f", median_total),
    sprintf("min_pseudobulk_fragment_counts\t%.0f", min(pseudobulk_totals)),
    sprintf("max_pseudobulk_fragment_counts\t%.0f", max(pseudobulk_totals)),
    sprintf("median_peak_total_counts\t%.0f", stats::median(Matrix::rowSums(counts))),
    sprintf("median_peak_length\t%.0f", stats::median(length_peak[with_sequence])),
    sprintf("median_peak_gc\t%.4f", stats::median(gc)),
    sprintf("pfm_counts_per_column\tmin=%.0f median=%.0f (MEME probabilities rescaled by nsites, before PFMatrix)",
            motif_qc$pfm_min_column_max, motif_qc$pfm_median_column_max),
    sprintf("pwm_distinct_values_per_motif\tmin=%.0f median=%.0f",
            motif_qc$pwm_min_distinct, motif_qc$pwm_median_distinct),
    sprintf("chromvar_seed\ts%d", seed),
    sprintf("background_peaks\tniterations=50 (chromVAR default), %d x %d", nrow(background), ncol(background)),
    sprintf("motif_match_threshold\tmotifmatchr defaults (p.cutoff=5e-05)"),
    sprintf("s03_min_cells_per_pseudobulk\t%s", scalars[["chromvar.min_cells_per_pseudobulk"]])
  )
  qc <- c(qc, sprintf("warning\t%s", if (length(warnings)) warnings else "none"))
  writeLines(qc, file.path(species_dir, "chromvar_qc.txt"))

  elapsed <- as.numeric(difftime(Sys.time(), started, units = "secs"))
  log_line(species, sprintf("done: %d motifs x %d pseudobulks, %d/%d peaks with sequence, %.1fs",
                            nrow(z), ncol(z), length(keep), length(peak_ids), elapsed))
  data.frame(species = species, status = "ok",
             n_peaks = length(peak_ids), n_peaks_with_sequence = length(keep_with_sequence),
             n_peaks_dropped_no_sequence = length(dropped_no_sequence),
             n_peaks_dropped_zero_counts = length(dropped_zero_counts),
             n_pseudobulk = length(pseudobulk_ids),
             n_motifs = length(pfms), n_motifs_matched = sum(motifs_with_match),
             n_peaks_with_motif = sum(peaks_with_motif),
             fraction_peaks_with_motif = round(fraction_with_motif, 6),
             n_na_z = n_na_z, fraction_na_z = round(fraction_na_z, 6),
             median_pseudobulk_counts = median_total,
             transposed_counts = loaded$transposed, seconds = round(elapsed, 1),
             note = if (length(warnings)) paste(warnings, collapse = " | ") else "")
}


# ---------------------------------------------------------------------- main

main <- function() {
  args <- parse_args(commandArgs(trailingOnly = TRUE))
  root <- project_root()
  config_path <- if (is.null(args$config)) file.path(root, DEFAULT_CONFIG) else args$config
  if (!file.exists(config_path)) stop("config not found: ", config_path, call. = FALSE)
  scalars <- config_scalars(config_path)

  results_root <- file.path(root, scalars[["paths.results"]])
  input_root <- if (is.null(args$input_root)) file.path(results_root, INPUT_STEP) else args$input_root
  out_root <- if (is.null(args$out_root)) file.path(results_root, CHROMVAR_STEP) else args$out_root
  dir.create(out_root, showWarnings = FALSE, recursive = TRUE)

  genome_table <- if (is.null(args$genome_table)) {
    file.path(root, "config", "species_genome.tsv")
  } else {
    args$genome_table
  }
  species_table <- read_species_table(genome_table, root)
  wanted <- if (is.null(args$species)) species_table$species else trimws(strsplit(args$species, ",")[[1]])
  unknown <- setdiff(wanted, species_table$species)
  if (length(unknown)) stop("unknown species: ", paste(unknown, collapse = ","), call. = FALSE)

  log_line("config", config_path, "| input", input_root, "| out", out_root)
  # The audit is flushed as each species finishes and one species' error is
  # turned into a status="error" row, so a late failure can never take the
  # record of the species that already succeeded (or were skipped) down with it.
  audit_path <- file.path(out_root, "chromvar_run_audit.tsv")
  written <- FALSE
  append_audit <- function(row) {
    write.table(row, audit_path, sep = "\t", quote = FALSE, row.names = FALSE,
                col.names = !written, append = written)
    written <<- TRUE
  }

  statuses <- character(0)
  for (species in wanted) {
    row <- tryCatch(
      run_species(species, species_table[species_table$species == species, , drop = FALSE][1, ],
                  scalars, input_root, out_root, root),
      error = function(error) {
        message <- conditionMessage(error)
        log_line("ERROR", species, message)
        blank_audit_row(species, "error", message)
      }
    )
    statuses <- c(statuses, as.character(row$status[[1]]))
    append_audit(row)
  }
  ok <- sum(statuses == "ok")
  log_line("finished", sprintf("%d/%d species ok", ok, length(statuses)))
  if (ok == 0L) quit(status = 1L)
}


main()
