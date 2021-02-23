FROM alpine:3.13 as dev
RUN apk add --no-cache rust cargo

COPY Cargo.toml .
COPY src src
RUN cargo build --release

FROM alpine:3.13 as base
RUN apk add --no-cache libgcc
RUN adduser -S user -G users
USER user
COPY --from=dev /target/release/p3216 /usr/bin/p3216
