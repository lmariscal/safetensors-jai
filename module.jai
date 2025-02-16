#import "Basic";
#import "File";
#import "String";
#import "Hash_Table";
#import,file "./modules/jaison/module.jai";

SafeTensors_Type :: enum u32 {
    UNKNOWN :: 0x00;
    BOOL;
    U8;
    I8;
    F8_E5M2;
    F8_E4M3;
    U16;
    I16;
    F16;
    BF16;
    U32;
    I32;
    F32;
    U64;
    I64;
    F64;
}

SafeTensors_Raw :: struct {
    file: string;
    header: string; // string_view
    body: string; // string_view
}

SafeTensors_Item :: struct {
    name: string;
    dtype: SafeTensors_Type;
    shape: [..]u32;
    data_offsets: [2]u64;
}

SafeTensor :: struct {
    raw: SafeTensors_Raw;
    items: [..]SafeTensors_Item;
}

load_safe_raw :: (file: string) -> result: SafeTensors_Raw = .{}, ok: bool = true {
    raw: SafeTensors_Raw = ---;
    ok: bool;
    raw.file, ok = read_entire_file(file);
    if !ok {
        log("Failed to read file '%", "asd");
        return ok = false;
    }

    raw.header.data = raw.file.data + size_of(u64);
    raw.header.count = xx cast(*u64, raw.file.data).*;

    raw.body.data = raw.file.data + size_of(u64) + raw.header.count;
    raw.body.count = raw.file.count - size_of(u64) + raw.header.count;

    return raw;
}

// recommended to pass the temp allocator, but providing a free function just in case
free :: (using safe: SafeTensor) {
    for items {
        free(it.name);
        array_free(it.shape);
    }
    array_free(safe.items);
    free(raw.file);
}

load_safe :: (file: string) -> result: SafeTensor = .{}, ok: bool = true {
    raw, ok := load_safe_raw(file);
    if !ok  return ok = false;

    result: SafeTensor = .{ raw = raw };

    ok=, json_data := json_parse_string(raw.header);
    if !ok {
        log("Failed parsing json header in SafeTensor '%'", file);
        return ok = false;
    }
    defer json_free(json_data);

    assert(json_data.type == .OBJECT, "Invalid SafeTensor Header 1");

    root_object := json_data.object;

    for item_object, item_name: root_object {
        array_add(*result.items, .{ name = copy_string(item_name) });
        item := *result.items[result.items.count-1];

        assert(item_object.type == .OBJECT, "Invalid SafeTensor Header 2");
        obj := item_object.object;

        assert(table_contains(obj, "dtype"), "Item in header doesn't contain 'dtype'");
        assert(table_contains(obj, "data_offsets"), "Item in header doesn't contain 'data_offses'");
        assert(table_contains(obj, "shape"), "Item in header doesn't contain 'shape'");

        dtype_value := table_find_pointer(obj, "dtype");
        assert(dtype_value.type == .STRING, "Invalid 'dtype' value");
        if dtype_value.str == {
            case "BOOL";
                item.dtype = .BOOL;
            case "U8";
                item.dtype = .U8;
            case "I8";
                item.dtype = .I8;
            case "F8_E5M2";
                item.dtype = .F8_E5M2;
            case "F8_E4M3";
                item.dtype = .F8_E4M3;
            case "U16";
                item.dtype = .U16;
            case "I16";
                item.dtype = .I16;
            case "F16";
                item.dtype = .F16;
            case "BF16";
                item.dtype = .BF16;
            case "U32";
                item.dtype = .U32;
            case "I32";
                item.dtype = .I32;
            case "F32";
                item.dtype = .F32;
            case "U64";
                item.dtype = .U64;
            case "I64";
                item.dtype = .I64;
            case "F64";
                item.dtype = .F64;
        }

        shape_value := table_find_pointer(obj, "shape");
        assert(shape_value.type == .ARRAY, "Invalid 'shape' value");
        for shape_value.array {
            assert(it.type == .NUMBER, "Invalid 'shape' value");
            array_add(*item.shape, xx it.number);
        }

        data_offsets_value := table_find_pointer(obj, "data_offsets");
        assert(data_offsets_value.type == .ARRAY, "Invalid 'data_offsets' value");
        x0 := data_offsets_value.array[0];
        x1 := data_offsets_value.array[1];
        assert(x0.type == .NUMBER, "Invalid 'data_offsets' value");
        assert(x1.type == .NUMBER, "Invalid 'data_offsets' value");
        item.data_offsets[0] = xx x0.number;
        item.data_offsets[1] = xx x1.number;
    }

    return result;
}

find_item :: inline (safe: *SafeTensor, name: string) -> *SafeTensors_Item {
    item: *SafeTensors_Item;
    for * safe.items {
        if it.name != name  continue;
        item = it;
        break;
    }
    return item;
}

// returns a slice of the data in the safetensor
view_item_data :: (safe: *SafeTensor, name: string, $type: Type) -> []type {
    // not expecting safetensors to contain many elements, so linear search should be okay
    item := find_item(safe, name);
    assert(item != null, tprint("Item '%' not found in SafeTensor", name));
    total_byte_size: s64 = xx (item.data_offsets[1] - item.data_offsets[0]);

    result: []type = ---;
    result.data = xx (safe.raw.body.data + item.data_offsets[0]);
    result.count = total_byte_size / size_of(type);

    return result;
}

// returns a copy of the data in the safetensor
get_item_data :: (safe: *SafeTensor, name: string, $type: Type) -> []type {
    // not expecting safetensors to contain many elements, so linear search should be okay
    item := find_item(safe, name);
    assert(item != null, tprint("Item '%' not found in SafeTensor", name));

    total_byte_size: s64 = xx (item.data_offsets[1] - item.data_offsets[0]);

    result := NewArray(total_byte_size / size_of(type), type, initialized = false);
    memcpy(result.data, safe.raw.body.data + item.data_offsets[0], total_byte_size);

    return result;
}
