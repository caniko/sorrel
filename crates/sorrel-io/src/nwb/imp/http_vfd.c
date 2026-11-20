#include <stdlib.h>
#include <hdf5.h>
#include <H5FDdevelop.h>

extern void sorrel_range_retain(const void *object);
extern void sorrel_range_release(const void *object);
extern int sorrel_range_read(const void *object, uint64_t addr, size_t size, void *buffer);

typedef struct {
    const void *object;
    uint64_t size;
} RangeAccess;

typedef struct {
    H5FD_t base;
    RangeAccess access;
    haddr_t eoa;
} RangeFile;

static void *copy_access(const void *ptr) {
    RangeAccess *copy = malloc(sizeof(*copy));
    if (copy) {
        *copy = *(const RangeAccess *)ptr;
        sorrel_range_retain(copy->object);
    }
    return copy;
}

static herr_t free_access(void *ptr) {
    sorrel_range_release(((RangeAccess *)ptr)->object);
    free(ptr);
    return 0;
}

static void *get_access(H5FD_t *file) {
    return copy_access(&((RangeFile *)file)->access);
}

static H5FD_t *open_range(const char *name, unsigned flags, hid_t fapl, haddr_t maxaddr) {
    const RangeAccess *access = H5Pget_driver_info(fapl);
    if (!name || flags != H5F_ACC_RDONLY || !access || access->size > maxaddr)
        return NULL;
    RangeFile *file = calloc(1, sizeof(*file));
    if (!file)
        return NULL;
    file->access = *access;
    sorrel_range_retain(access->object);
    return &file->base;
}

static herr_t close_range(H5FD_t *file) {
    sorrel_range_release(((RangeFile *)file)->access.object);
    free(file);
    return 0;
}

static haddr_t get_eoa(const H5FD_t *file, H5FD_mem_t type) {
    (void)type;
    return ((const RangeFile *)file)->eoa;
}

static herr_t set_eoa(H5FD_t *file, H5FD_mem_t type, haddr_t addr) {
    (void)type;
    ((RangeFile *)file)->eoa = addr;
    return 0;
}

static haddr_t get_eof(const H5FD_t *file, H5FD_mem_t type) {
    (void)type;
    return ((const RangeFile *)file)->access.size;
}

static herr_t read_range(H5FD_t *file, H5FD_mem_t type, hid_t dxpl, haddr_t addr, size_t size, void *buffer) {
    (void)type; (void)dxpl;
    return sorrel_range_read(((RangeFile *)file)->access.object, addr, size, buffer);
}

static herr_t reject_write(H5FD_t *file, H5FD_mem_t type, hid_t dxpl, haddr_t addr, size_t size, const void *buffer) {
    (void)file; (void)type; (void)dxpl; (void)addr; (void)size; (void)buffer;
    return -1;
}

// Use the complete class from the same headers as the linked HDF5, not the
// partial Rust binding. Unimplemented capabilities are explicitly null.
static const H5FD_class_t range_class = {
    .version = H5FD_CLASS_VERSION,
    .value = H5_VFD_RESERVED,
    .name = "sorrel_range",
    .maxaddr = HADDR_MAX,
    .fc_degree = H5F_CLOSE_WEAK,
    .terminate = NULL,
    .sb_size = NULL,
    .sb_encode = NULL,
    .sb_decode = NULL,
    .fapl_size = sizeof(RangeAccess),
    .fapl_get = get_access,
    .fapl_copy = copy_access,
    .fapl_free = free_access,
    .dxpl_size = 0,
    .dxpl_copy = NULL,
    .dxpl_free = NULL,
    .open = open_range,
    .close = close_range,
    .cmp = NULL,
    .query = NULL,
    .get_type_map = NULL,
    .alloc = NULL,
    .free = NULL,
    .get_eoa = get_eoa,
    .set_eoa = set_eoa,
    .get_eof = get_eof,
    .get_handle = NULL,
    .read = read_range,
    .write = reject_write,
    .read_vector = NULL,
    .write_vector = NULL,
    .read_selection = NULL,
    .write_selection = NULL,
    .flush = NULL,
    .truncate = NULL,
    .lock = NULL,
    .unlock = NULL,
    .del = NULL,
    .ctl = NULL,
    .fl_map = H5FD_FLMAP_SINGLE,
};

hid_t sorrel_range_open(const void *object, uint64_t size) {
    // The Rust caller holds hdf5-metno's lock, including on threadsafe builds.
    static hid_t driver = H5I_INVALID_HID;
    if (driver < 0)
        driver = H5FDregister(&range_class);
    if (driver < 0)
        return H5I_INVALID_HID;
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    if (fapl < 0)
        return H5I_INVALID_HID;
    RangeAccess access = {object, size};
    hid_t file = H5I_INVALID_HID;
    if (H5Pset_driver(fapl, driver, &access) >= 0)
        file = H5Fopen("sorrel-remote-nwb", H5F_ACC_RDONLY, fapl);
    hid_t errors = file < 0 ? H5Eget_current_stack() : H5I_INVALID_HID;
    H5Pclose(fapl);
    if (errors >= 0)
        H5Eset_current_stack(errors);
    return file;
}

void sorrel_range_error(const char *message) {
    H5Epush2(H5E_DEFAULT, __FILE__, "read_range", __LINE__, H5E_ERR_CLS, H5E_VFL, H5E_READERROR, "%s", message);
}
