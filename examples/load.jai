#import "Basic";
#import,file "../module.jai";

main :: () {
    tensor, ok := load_safe("sample.safetensors",, temp);

    log("%", tensor.raw.header);
    a := get_item_data(*tensor, "a", float32);
    defer array_free(a);

    log("%", a);
}
