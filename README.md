# leteasyai-doc
leteasyai 文档 仓库





## build before
You can try one of these:

1. Downgrade to Node.js v16.

You can reinstall the current LTS version from Node.js’ website.

You can also use nvm. For Windows, use nvm-windows.

2. Enable legacy OpenSSL provider.

On Unix-like (Linux, macOS, Git bash, etc.):

```shell
export NODE_OPTIONS=--openssl-legacy-provider
```
On Windows command prompt:
```shell
set NODE_OPTIONS=--openssl-legacy-provider
```
On PowerShell:
```shell
$env:NODE_OPTIONS = "--openssl-legacy-provider"
```