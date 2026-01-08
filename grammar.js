/**
 * @file python parser
 * @author fnoble <felix.noble@neuro-genesis.uk>
 * @license MIT
 */

/// <reference types="tree-sitter-cli/dsl" />
// @ts-check

module.exports = grammar({
  name: "python",

  rules: {
    // TODO: add the actual grammar rules
    source_file: $ => "hello"
  }
});
