{
  description = "A lightweight re-implementation of the InstructOR transformer models";
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/a983cc62cc2345443620597697015c1c9c4e5b06";
    utils.url = "github:numtide/flake-utils/93a2b84fc4b70d9e089d029deacc3583435c2ed6";
  };
  outputs =
    { self
    , nixpkgs
    , utils
    ,
    }:
    let
      out = system:
        let
          pkgs = import nixpkgs {
            inherit system;
            config = { allowUnfree = true; };
          };
          inherit (pkgs) poetry2nix stdenv lib;
          overrides = pyfinal: pyprev: rec {
            torch =
              if system == "aarch64-darwin"
              then
                pyprev.pytorch-bin.overridePythonAttrs
                  (old: {
                    src = pkgs.fetchurl {
                      url = "https://download.pytorch.org/whl/cpu/torch-1.13.1-cp39-none-macosx_11_0_arm64.whl";
                      sha256 = "sha256-4N+QKnx91seVaYUy7llwzomGcmJWNdiF6t6ZduWgSUk=";
                    };
                  })
              else pyprev.torch;

            sentence-transformers =
              pyprev.sentence-transformers.overridePythonAttrs
                (old: {
                  buildInputs = (old.buildInputs or [ ]) ++ [ pyfinal.setuptools ];
                });
          };
          poetryEnv = pkgs.poetry2nix.mkPoetryEnv {
            python = pkgs.python39;
            projectDir = ./.;
            preferWheels = true;
            overrides = poetry2nix.overrides.withDefaults overrides;
            groups = [ "dev" "test" ];
          };
        in
        {
          devShell = pkgs.mkShell {
            buildInputs = with pkgs; [
              poetry
              poetryEnv
            ];
            PYTHONBREAKPOINT = "ipdb.set_trace";
          };
        };
    in
    utils.lib.eachDefaultSystem out;
}
